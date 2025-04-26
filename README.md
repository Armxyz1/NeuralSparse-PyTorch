# NeuralSparse-PyTorch

This repository provides a PyTorch implementation of the Graph Sparsification technique from the paper [Robust Graph Representation Learning via Neural Sparsification](https://openreview.net/forum?id=S1emOTNKvS). The implementation is applied to the OGBN-Proteins dataset.

NOTE: For this implementation, I've sampled edges from immediate neighbors and not 1-hop neighbors due to the needs of the dataset (edge embeddings aren't available for every edge).

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To train the model using graph sparsification, execute:

```bash
python main.py
```

To train the model without graph sparsification, execute:

```bash
python main.py --mode normal
```

## Results

The table below shows the performance of the model with and without graph sparsification:

| **Model**           | **ROC-AUC** |
|-----------------|---------|
| Normal          | 0.7013 |
| Sparsification  | 0.7269  |

## Weights

The weights of the sparsified model are saved in the `sparse_wts` directory. The weights of the normal model are saved in the `normal_wts` directory.
