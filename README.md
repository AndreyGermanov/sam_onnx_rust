# Segment anything inference using Rust

This Web Application demonstrates how to use Segment Anything neural network model exported to ONNX to extract the object from the image and remove background around it.

This repository includes small Mobile SAM model files, exported to ONNX: `vit_t_encoder.onnx` and `vit_t_decoder.onnx`. I can't upload bigger files to GitHub, so if you want to use bigger SAM models to get better accuracy, read my article [Export Segment Anything neural network to ONNX: the missing parts](https://dev.to/andreygermanov/export-segment-anything-neural-network-to-onnx-the-missing-parts-43c8) to learn how to export different models and download [Jupyter Notebook](https://github.com/AndreyGermanov/sam_onnx_full_export/blob/main/sam_onnx_export.ipynb) to run this procedure.

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/sam_onnx_rust.git`
* Ensure that the ONNX runtime installed on your operating system, because the library that integrated to the
Rust package may not work correctly. To install it, you can download the archive for your operating system
from [here](https://github.com/microsoft/onnxruntime/releases), extract and copy contents of "lib" subfolder
to the system libraries path of your operating system.

## Run

Execute:

```
cargo run
```

It will download and install all dependecies first and then start a webserver on http://localhost:8080. Use any web browser to open the web interface.

The animation above shows how the web interface works.
