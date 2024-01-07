use std::sync::Arc;
use image::{GenericImage, GenericImageView, Rgba};
use image::imageops::FilterType;
use ndarray::{Array};
use ort::{Environment, SessionBuilder, Value};

fn main() {

    // Load Segment Anything Encoder and Decoder models
    let env = Arc::new(Environment::builder().with_name("SAM").build().unwrap());
    let encoder = SessionBuilder::new(&env).unwrap().with_model_from_file("vit_t_encoder.onnx").unwrap();
    let decoder = SessionBuilder::new(&env).unwrap().with_model_from_file("vit_t_decoder.onnx").unwrap();

    // Open and encode the input image to (1,3,1024,1024) tensor
    let img = image::open("cat_dog.jpg").expect("Could not open image");
    let (orig_width, orig_height) = (img.width() as f32,img.height() as f32);
    println!("Original size: {},{}",orig_width, orig_height);
    let img_resized = img.resize(1024, 1024, FilterType::CatmullRom);
    let(resized_width, resized_height) = (img_resized.width() as f32, img_resized.height() as f32);
    println!("Resized size: {},{}", resized_width, resized_height);
    let mut input = Array::zeros((1,3, 1024, 1024)).into_dyn();

    let mean = vec![123.675, 116.28, 103.53];
    let std = vec![58.395, 57.12, 57.375];
    for pixel in img_resized.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r,g,b,_] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32 - mean[0]) / std[0];
        input[[0, 1, y, x]] = (g as f32 - mean[1]) / std[1];
        input[[0, 2, y, x]] = (b as f32 - mean[2]) / std[2];
    };
    let input_as_values = &input.as_standard_layout();
    let encoder_inputs = vec![Value::from_array(encoder.allocator(), input_as_values).unwrap()];
    println!("Input tensor shape: {:?}",input.shape());

    // Run encoder to get image embeddings
    let mut outputs = encoder.run(encoder_inputs).unwrap();
    let embeddings = outputs.get(0)
        .unwrap().try_extract::<f32>().unwrap().view().t().reversed_axes().into_owned();

    println!("Embeddings shape: {:?}",&embeddings.shape());

    // Prepare input for decoder

    // Encode image embeddings
    let embeddings_as_values = &embeddings.as_standard_layout();

    // Encode points prompt
    let (x,y) = (321.0, 230.0);
    let point_coords = Array::from_shape_vec((1,2,2),
     vec![x*(resized_width/orig_width),y*(resized_height/orig_height),0.0,0.0])
        .unwrap().into_dyn().into_owned();
    let point_coords_as_values = &point_coords.as_standard_layout();
    let point_labels = Array::from_shape_vec((1,2),vec![1.0_f32,-1.0_f32])
        .unwrap().into_dyn().into_owned();
    let point_labels_as_values = &point_labels.as_standard_layout();

    // Encode mask prompt
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

    // Prepare inputs for decoder
    let decoder_inputs = vec![
        Value::from_array(encoder.allocator(), embeddings_as_values).unwrap(),
        Value::from_array(encoder.allocator(), point_coords_as_values).unwrap(),
        Value::from_array(encoder.allocator(), point_labels_as_values).unwrap(),
        Value::from_array(encoder.allocator(), mask_input_as_values).unwrap(),
        Value::from_array(encoder.allocator(), has_mask_input_as_values).unwrap(),
        Value::from_array(encoder.allocator(), orig_im_size_as_values).unwrap(),
    ];

    // Run the decoder
    outputs = decoder.run(decoder_inputs).unwrap();
    let output = outputs.get(0).unwrap().try_extract::<f32>()
        .unwrap().view().t().reversed_axes().into_dyn().into_owned();

    println!("Mask shape: {:?}", output.shape());

    // Process output mask
    let mut mask_img = image::DynamicImage::new_rgb8(orig_width as u32, orig_height as u32);
    let mut index = 0.0;
    output.for_each(|item| {
        let color = if *item > 0.0 { Rgba::<u8>([255,255,255,1])  } else { Rgba::<u8>([0,0,0,1]) };
        let y = f32::floor(index / orig_width);
        let x = index - y * orig_width;
        mask_img.put_pixel(x as u32, y as u32, color);
        index += 1.0;
    });

    // Save mask to a file
    mask_img.save("mask.png").expect("Could not save mask to a file");
}
