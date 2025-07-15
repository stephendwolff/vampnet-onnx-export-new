# VampNet (ONNX version)

This repository provides tools for exporting the original VampNet models to ONNX format and running inference with the ONNX models while maintaining the same interface as the original VampNet implementation. The goal is to enable cross-platform deployment and improved inference performance through ONNX optimization.

The repository includes `hello_onnx.py` which replicates the functionality of the original VampNet repository's `hello.py` script, demonstrating audio generation using the ONNX models. While encoding and decoding work correctly in isolation, there are ongoing efforts to match the final audio quality of the original implementation.

## Getting Started

To use this repository, you must first export the VampNet models to ONNX format by following the instructions in [ONNX_EXPORT.md](./ONNX_EXPORT.md). 

The export process includes:
1. Setting up a Python 3.11 environment
2. Downloading the original VampNet models from Zenodo
3. Running the export scripts to generate ONNX models
4. The exported models will be placed in the `models_onnx` directory

Once the models are exported, you can run inference using `hello_onnx.py` or integrate the ONNX models into your own applications.
