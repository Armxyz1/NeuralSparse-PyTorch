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

The table below shows the performance of the model with various sparsification and sampling methods. The results are based on the ROC AUC metric, which is a common evaluation metric for graph representation learning tasks.

**TABLE I**: System parameters while training
| **Sparsification** <br> **Method** | **Sampling** <br> **Method** | **CPU** <br> **Usage (%)** | **CPU** <br> **Power (W)** | **GPU** <br> **Usage (%)** | **GPU** <br> **Power (W)** | **RAM** <br> **(MB)** | **vRAM** <br> **(MB)** |
|:----------------------------------:|:----------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:---------------------:|:----------------------:|
| None               | Uniformly Random | 29.6 | 10.2 | 6.3 | 4.3 | 4212 | 1450 |
| None               | GraphSAGE        | 40.9 | 18.6 | 15.6 | 5.0 | 6481 | 433 |
| None               | GraphSAINT       | 45.0 | 20.1 | 23.7 | 9.8 | 7121 | 1055 |
| DropEdge           | Uniformly Random | 40.3 | 18.0 | 42.8 | 20.0 | 3141 | 3388 |
| DropEdge           | GraphSAGE        | 44.0 | 19.5 | 41.8 | 19.3 | 6778 | 2560 |
| DropEdge           | GraphSAINT       | 47.2 | 21.2 | 23.9 | 9.3 | 7391 | 849 |
| NeuralSparse       | Uniformly Random | 36.5 | 14.2 | 27.4 | 16.4 | 3074 | 1639 |
| NeuralSparse       | GraphSAGE        | 38.9 | 16.0 | 24.9 | 13.4 | 6802 | 2955 |
| NeuralSparse       | GraphSAINT       | 36.8 | 14.6 | 22.9 | 9.1 | 7713 | 1943 |


**TABLE II**: Training results
| **Sparsification** <br> **Method** | **Sampling** <br> **Method** | **Time per** <br> **Epoch (s)** | **Number of** <br> **Epochs** | **ROC** <br> **AUC (%)** | **EDP (W min²)** |
|:----------------------------------:|:----------------------------:|:---------------------------:|:----------------------------:|:--------------------:|:----------------:|
| None               | Uniformly Random | 18.1 | 49 | 66.56 | 3605.21 |
| None               | GraphSAGE        | 14.7 | 29 | 68.27 | 1726.45 |
| None               | GraphSAINT       | 1.4  | 45 | 72.23 | 48.29 |
| DropEdge           | Uniformly Random | 36.4 | 17 | 64.15 | 6413.80 |
| DropEdge           | GraphSAGE        | 28.2 | 38 | 74.76 | 19553.45 |
| DropEdge           | GraphSAINT       | 1.8  | 31 | 70.53 | **39.01** |
| NeuralSparse       | Uniformly Random | 106.2| 25 | 76.46 | 81455.40 |
| NeuralSparse       | GraphSAGE        | 84.1 | 19 | 77.34 | 29008.15 |
| NeuralSparse       | GraphSAINT       | 57.1 | 37 | **79.17** | 46494.81 |

