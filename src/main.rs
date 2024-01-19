use std::sync::{Arc, Mutex};
use std::path::Path;
use image::{GenericImage, GenericImageView, Rgba};
use image::imageops::FilterType;
use ndarray::{Array, ArrayBase, AssignElem, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use rocket::{response::content, fs::TempFile, form::Form, error, State};
use rocket::http::Status;

#[macro_use] extern crate rocket;

// Web application state
struct AppState {
    image_embeddings: Mutex<Array<f32,IxDyn>>,
    orig_width: Mutex<f32>,
    orig_height: Mutex<f32>,
    resized_width: Mutex<f32>,
    resized_height: Mutex<f32>,
    encoder: Session,
    decoder: Session
}

// Main function that defines
// a web service endpoints a starts
// the web service
#[rocket::main]
async fn main() {
    // Load ONNX models for Mobile SAM encoder and decoder and store it to the Web Application state
    let env = Arc::new(Environment::builder().with_name("SAM").build().unwrap());
    let encoder = SessionBuilder::new(&env).unwrap().with_model_from_file("vit_t_encoder.onnx").unwrap();
    let decoder = SessionBuilder::new(&env).unwrap().with_model_from_file("vit_t_decoder.onnx").unwrap();

    // Start Rocket web server, bind state and endpoints to it
    rocket::build()
        .manage(AppState {
            image_embeddings: Mutex::new(ArrayBase::zeros((1, 256, 64, 64)).into_dyn()),
            orig_width: Mutex::new(0_f32),
            orig_height: Mutex::new(0_f32),
            resized_width: Mutex::new(0_f32),
            resized_height: Mutex::new(0_f32),
            encoder: encoder,
            decoder: decoder
        })
        .mount("/", routes![index])
        .mount("/encode", routes![encode])
        .mount("/decode", routes![decode])
        .launch().await.unwrap();
}

// Site main page handler function.
// Returns Content of index.html file
#[get("/")]
fn index() -> content::RawHtml<String> {
    content::RawHtml(std::fs::read_to_string("index.html").unwrap())
}

// Handler of /encode POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through SAM encoder to get image embeddings.
//
// Returns the status of operation
#[post("/", data = "<file>")]
fn encode(file: Form<TempFile<'_>>, state: &State<AppState>) -> Status {
    let result = std::fs::read(file.path().unwrap_or(Path::new(""))).unwrap();
    let encoder = &state.encoder;
    let (embeddings, orig_width, orig_height, resized_width, resized_height) = get_image_embeddings(result, &encoder).unwrap();
    *state.image_embeddings.lock().unwrap() = embeddings;
    *state.orig_width.lock().unwrap() = orig_width;
    *state.orig_height.lock().unwrap() = orig_height;
    *state.resized_width.lock().unwrap() = resized_width;
    *state.resized_height.lock().unwrap() = resized_height;
    return Status::Ok
}

// Structure used to receive POST form data
// with box prompt coordinages [x1,y1,x2,y2]
#[derive(FromForm)]
struct Coords {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32
}

#[post("/", data = "<coords>")]
// Handler of /decode POST endpoint
// Receives box prompt coordinates as a "Coords" struct,
// then builds SAM prompt from it, and, using previously
// stored image embeddings and information about image size,
// runs the SAM decoder model to get segmentation mask.
// returns segmentation mask as a flat array in which "0" are
// background pixels and "1" are foreground pixels.
fn decode(coords: Option<Form<Coords>>, state: &State<AppState>) -> String {
    let coords = coords.unwrap();
    let mask = decode_prompt(coords.x1, coords.y1, coords.x2, coords.y2, state);
    return serde_json::to_string(&mask).unwrap_or_default()
}

// Method used to encode image embeddings from loaded image using SAM encoder
// @param buf - image
// @param encoder - SAM encoder ONNX model instance
// @returns image embeddings
fn get_image_embeddings(buf: Vec<u8>, encoder: &Session) -> Option<(Array<f32,IxDyn>,f32,f32,f32,f32)> {
    // Load and encode the input image to (1,3,1024,1024) tensor
    let img = image::load_from_memory(&buf).ok()?;

    // Resize the image and preserve original size.
    let (orig_width, orig_height) = (img.width() as f32, img.height() as f32);
    let img_resized = img.resize(1024, 1024, FilterType::CatmullRom);
    let (resized_width, resized_height) = (img_resized.width() as f32, img_resized.height() as f32);

    // Copy the image pixels to the tensor, normalizing them using "mean" and "std" deviation
    let mut input = Array::zeros((1, 3, 1024, 1024)).into_dyn();
    let mean = vec![123.675, 116.28, 103.53];
    let std = vec![58.395, 57.12, 57.375];
    for pixel in img_resized.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32 - mean[0]) / std[0];
        input[[0, 1, y, x]] = (g as f32 - mean[1]) / std[1];
        input[[0, 2, y, x]] = (b as f32 - mean[2]) / std[2];
    };

    // Prepare tensor for the SAM encoder model
    let input_as_values = &input.as_standard_layout();
    let encoder_inputs = vec![Value::from_array(encoder.allocator(), input_as_values).ok()?];

    // Run encoder to get image embeddings
    let outputs = encoder.run(encoder_inputs).ok()?;
    let embeddings = outputs.get(0)?.try_extract::<f32>().ok()?.view().t().reversed_axes().into_owned();

    return Some((embeddings,orig_width,orig_height,resized_width,resized_height))
}

// Method used to get segmentation mask of object, inside x1,y1,x2,y2 box
// using previously encoded image embeddings from application state
// @param x1,y1,x2,y2 - coordinates of the box
// @param state - Rocket Web application state
fn decode_prompt(x1:f32, y1:f32, x2:f32, y2:f32,state: &State<AppState>) -> Vec<u8> {
    // Prepare input for decoder

    // Get embeddings, image sizes and ONNX model instances from Web Application state
    let embeddings = &state.image_embeddings.lock().unwrap().clone();
    let embeddings_as_values = &embeddings.as_standard_layout();
    let orig_width = state.orig_width.lock().unwrap().abs();
    let orig_height = state.orig_height.lock().unwrap().abs();
    let resized_width = state.resized_width.lock().unwrap().abs();
    let resized_height = state.resized_height.lock().unwrap().abs();
    let decoder = &state.decoder;

    // Encode points prompt
    let point_coords = Array::from_shape_vec((1,2,2),
     vec![x1*(resized_width/orig_width),y1*(resized_height/orig_height),
          x2*(resized_height/orig_height),y2*(resized_height/orig_height)])
        .unwrap().into_dyn().into_owned();
    let point_coords_as_values = &point_coords.as_standard_layout();
    let point_labels = Array::from_shape_vec((1,2),vec![2.0_f32,3.0_f32])
        .unwrap().into_dyn().into_owned();
    let point_labels_as_values = &point_labels.as_standard_layout();

    // Encode mask prompt (dummy)
    let mask_input = Array::from_shape_vec((1,1,256,256),
       vec![0;256*256].iter().map(|v| *v as f32).collect())
        .unwrap().into_dyn().into_owned();
    let mask_input_as_values = &mask_input.as_standard_layout();
    let has_mask_input = Array::from_vec(vec![0.0_f32]).into_dyn().into_owned();
    let has_mask_input_as_values = &has_mask_input.as_standard_layout();

    // Add original image size
    let orig_im_size = Array::from_vec(vec![orig_height,orig_width])
        .into_dyn().into_owned();
    let orig_im_size_as_values = &orig_im_size.as_standard_layout();

    // Prepare inputs for SAM decoder
    let decoder_inputs = vec![
        Value::from_array(decoder.allocator(), embeddings_as_values).unwrap(),
        Value::from_array(decoder.allocator(), point_coords_as_values).unwrap(),
        Value::from_array(decoder.allocator(), point_labels_as_values).unwrap(),
        Value::from_array(decoder.allocator(), mask_input_as_values).unwrap(),
        Value::from_array(decoder.allocator(), has_mask_input_as_values).unwrap(),
        Value::from_array(decoder.allocator(), orig_im_size_as_values).unwrap(),
    ];

    // Run the SAM decoder
    let outputs = decoder.run(decoder_inputs).unwrap();
    let output = outputs.get(0).unwrap().try_extract::<f32>()
        .unwrap().view().t().reversed_axes().into_dyn().into_owned();

    // Process and return output mask (replace negative pixel values to 0 and positive to 1)
    return output.map(|item|  if *item > 0.0 { 1_u8 } else {0_u8}).into_raw_vec()
}
