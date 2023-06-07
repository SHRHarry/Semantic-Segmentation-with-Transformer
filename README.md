# Semantic-Segmentation-with-Transformer

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/docs/transformers/model_doc/segformer)
[![paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2105.15203)
## Introduction

A repository of SegFormer model test

## Architecture

<p align="center">
<img src="fig/segformer_architecture.png" width = "747" height = "380" alt="segformer architecture" />
</p>

## Experimentals Results

<p align="center">
<img src="fig/result-1.png" width = "350" height = "350" alt="result-1" /><img src="fig/result-2.png" width = "350" height = "350" alt="result-2" />
</p>

## Installation

Assuming a fresh Anaconda distribution with Python 3.8, you can install the dependencies with:

```sh
cd Semantic-Segmentation-with-Transformer
pip install -r requirements.txt
```

## Training

```sh
python semantic_segmentation_main.py --api train
```

## Evaluate

```sh
python semantic_segmentation_main.py --api eval
```

## Inference

```sh
python semantic_segmentation_main.py --api infer
```

## References

+ https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb