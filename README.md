# Sudoku Autoregressive Transformer — Baseline 2

Part of a project comparing autoregressive and diffusion-based approaches to constraint satisfaction problems.

---

## What This Is

A standard autoregressive transformer trained to solve Sudoku puzzles token by token, left to right. No constraint rules are specified — only data.

---

## Architecture

| Component             | Value
|---                    |---
| Sequence length       | 81 tokens (one per cell, row-major)
| Vocabulary            | 10 tokens (digits 0–9, where 0 = blank)
| Embedding dim         | 128
| Attention heads       | 4
| Transformer layers    | 4
| Feedforward dim       | 512
| Output                | Linear(128, 10) → argmax
| Total parameters      | 1,071,242

The model uses a transformer decoder with a causal mask. Each cell attends only to cells to its left. Position is encoded by adding learned position embeddings to token embeddings before the transformer layers.


### Visualization
```
raw integers:     [0, 0, 3, 0, 2, ...]         shape [81]
after embedding:  [[0.2,-0.5,...], ...]        shape [81, 128]
after + position: [[0.3,-0.4,...], ...]        shape [81, 128]  ← combined
after transformer:[[0.7, 0.1,...], ...]        shape [81, 128]  ← context added
after linear:     [[0.1,0.05,0.6,...], ...]    shape [81, 10]   ← 10 scores

PUZZLE                    MODEL                         LOSS
                    
[0,0,3,0,2,...]     each integer                   compare predictions
81 integers    →    becomes a                  →   to correct answers  →  2.34
                    128-dim vector                  
                    ↓                               
                    transformer mixes               
                    vectors together                
                    ↓                               
                    each vector becomes             
                    10 scores                       
                    ↓                               
                    [64, 81, 10]

LOSS NUMBER          BACKWARD PASS              WEIGHT UPDATE

2.34            →    how did each of        →   nudge each weight
                     1M weights contribute      slightly toward
                     to this 2.34?              lower loss
                     
                     result: 1M gradients
```

---

## Training

```
Dataset:   Kaggle 1M Sudoku dataset (bryanpark/sudoku)
           500,000 puzzles used for training
Epochs:    20
Batch:     64
Optimizer: Adam, lr=1e-3
Loss:      CrossEntropyLoss
Device:    Apple MPS (M1-series Mac)
Time:      ~9 hours
```

The dataset is uniformly easy — all puzzles have 31–36 givens (avg 33.8). No hard puzzles in the training distribution.

---

## Results

**Easy puzzles (31–36 givens, n=500,000):**

```
Cell accuracy:   96.31%
Puzzle accuracy: 11.88%
```

**Accuracy by board position** — left to right, top to bottom:

```
85.6 | 88.9 | 90.0 | 91.4 | 92.3 | 93.2 | 93.3 | 93.9 | 94.7
91.7 | 92.9 | 93.9 | 93.6 | 94.7 | 95.2 | 95.2 | 95.8 | 96.4
94.2 | 95.2 | 96.0 | 95.7 | 96.4 | 96.9 | 96.2 | 96.9 | 97.3
92.2 | 93.3 | 94.1 | 95.0 | 95.7 | 96.3 | 96.6 | 97.0 | 97.4
94.8 | 95.7 | 96.5 | 96.5 | 97.3 | 97.7 | 97.7 | 98.1 | 98.4
96.5 | 97.3 | 97.8 | 97.7 | 98.2 | 98.6 | 98.3 | 98.7 | 98.9
94.9 | 95.7 | 96.2 | 97.1 | 97.7 | 98.0 | 98.4 | 98.7 | 98.9
96.6 | 97.2 | 97.8 | 98.1 | 98.5 | 98.8 | 99.0 | 99.3 | 99.4
97.8 | 98.3 | 98.8 | 99.0 | 99.2 | 99.4 | 99.4 | 99.5 | 99.6
```

The gradient from A1 (85.6%) to I9 (99.6%) is direct evidence of the autoregressive weakness. Early cells are predicted with minimal context, late cells benefit from prior predictions.

**Hard puzzle (17 givens — hard1 from Norvig):**

The model produces an invalid solution with duplicate digits in rows, columns, and boxes. Completely out of distribution. Trained exclusively on 31–36 given puzzles.
