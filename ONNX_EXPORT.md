# VampNet ONNX Export

This document explains how to export the original VampNet models to ONNX for use with vampnet_onnx.

A bash script `export_all.sh` can be run, after the models have been downloaded as described as the first of the set up steps.

## Set up

First the original models need to be downloaded and placed in the `/models` directory (as per the original vampnet).

```
models/vampnet/c2f.pth
models/vampnet/coarse.pth
models/vampnet/codec.pth
models/wavebeat.pth
```

These can be downloaded from https://zenodo.org/records/8136629 (link from bottom of original Vampnet README: https://github.com/hugofloresgarcia/vampnet?tab=readme-ov-file#take-a-look-at-the-pretrained-models)

A Python3.11 virtual environment should be set up and activated with the `requirements.txt`

```bash
$ pyenv local 3.11
$ python -m venv venv
$ pip install -r requirements.txt
```

## Running the export scripts



1. Codec, `lac_encoder.onnx` and `lac_decoder.onnx` files:

```
python scripts/codec_onnx_export.py
```

2. Codebooks, `lac_from_codes.onnx` and `lac_codebook_tables.pth`:

```
python scripts/export_from_codes_with_proj.py
python scripts/codec_onnx_extract_codebooks.py
```

3. Quantizer `lac_quantizer.onnx`

```
python scripts/codec_onnx_export_quantizer.py
```

4. Coarse embeddings and transformer

```
python scripts/coarse_onnx_export.py
```


5. Coarse to fine (c2f) embeddings and transformer

```
python scripts/c2f_onnx_export.py
```