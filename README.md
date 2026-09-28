# GraphRNN

This repository implements a GraphRNN-based approach for learning and generating graph structures from autoregressive node sequences.

## Overview

The codebase provides a complete training and generation pipeline for graph generation, including:

- synthetic graph dataset generation (`community`, `grid`, `ego`)
- graph sequence preparation and batching
- GraphRNN model training
- graph sampling from trained checkpoints
- evaluation using MMD over degree and clustering distributions

## Repository structure

```text
GraphRNN/
├── data/
│   ├── processed/
│   └── raw/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── results/
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── train.py
│   └── model/
│       ├── model.py
│       └── utils.py
├── requirements.txt
├── README.md
└── notebooks/
```

## Requirements

- Python 3.10+
- pip
- CUDA is optional if GPU acceleration is available

## Installation

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the model on the `community` dataset:

```bash
python -m src.train --dataset community --variant rnn --epochs 5 --batch_size 32
```

Additional examples:

```bash
python -m src.train --dataset grid --variant s --epochs 10 --lr 0.001
python -m src.train --dataset ego --variant rnn --epochs 5
```

Checkpoints are saved to:

```text
outputs/checkpoints/
```

## Generation

Generate graphs from a trained checkpoint:

```bash
python -m src.generate --checkpoint outputs/checkpoints/community_rnn_epoch_5.pt --num_graphs 10
```

Generated graphs are saved in:

```text
outputs/results/
```

## Evaluation

Evaluate generated graphs against the test set:

```bash
python -m src.evaluate --dataset community --variant rnn
```

The evaluation metrics are stored in JSON format under:

```text
outputs/results/
```

## Output directories

- `outputs/checkpoints/` : model checkpoints
- `outputs/figures/` : training curves and generated graph visualizations
- `outputs/results/` : generated graph files and evaluation metrics

## Notes

- The training pipeline reads from `data/processed` and creates datasets automatically if they do not exist.
- The `--variant` argument supports:
  - `rnn` : full recurrent GraphRNN variant
  - `s` : simplified variant
- The model is designed to generate structured graphs from sequential adjacency representations.

## Quick example

```bash
python -m src.train --dataset community --variant rnn --epochs 3
python -m src.generate --checkpoint outputs/checkpoints/community_rnn_epoch_3.pt --num_graphs 5
python -m src.evaluate --dataset community --variant rnn
```

## License

This repository is provided for research and experimentation purposes.
