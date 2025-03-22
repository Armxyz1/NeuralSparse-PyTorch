# NeuralSparse-PyTorch

This repository provides a PyTorch implementation of the Graph Sparsification technique from the paper [Robust Graph Representation Learning via Neural Sparsification](https://openreview.net/forum?id=S1emOTNKvS). The implementation is applied to the OGBN-Proteins dataset.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train the model using graph sparsification, execute the following command:

```bash
python main.py
```

To train the model without graph sparsification, execute the following command:

```bash
python main.py --mode normal
```