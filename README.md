# Autoregressive Transformer — Architecture Notes

## The Research Context

This model is **Baseline 2** in a research project comparing approaches to solving constraint satisfaction problems. The research question is whether diffusion models can find better approximate solutions to NP-complete problems than autoregressive models within a fixed compute budget. Sudoku is the tractable proxy. Protein folding is the ultimate target.

**Baseline 1** is Norvig's classical constraint propagation + backtracking solver. It solves easy puzzles in ~2ms with 100% accuracy. It solves hard puzzles (17 givens, 64 ambiguous cells after propagation) in ~30 seconds. It requires the constraint rules to be hand-coded explicitly.

**Baseline 2** (this model) is an autoregressive transformer. It learns to solve Sudoku from data, with no rules specified in advance. Its fundamental weakness — early commitment without global information — is the thing the diffusion model is designed to fix.

---

## The Task

Given a puzzle string of 81 characters (`'003020600...'`, where `0` means blank), predict the full solution string of 81 digits.

This is framed as **sequence-to-sequence classification**:

- Input: 81 tokens (the puzzle, with 0s for blanks)
- Output: 81 tokens (the solution, all digits filled in)
- At each position, predict one of 10 possible digits (0–9)

---

## Encoding

A Sudoku puzzle is represented as a sequence of 81 integers:

```python
puzzle   = '003020600...'
encoded  = [0, 0, 3, 0, 2, 0, 6, 0, 0, ...]  # length 81
```

The full training example is a pair of tensors:

```
X shape: [batch_size, 81]  — puzzle tokens (integers 0–9)
Y shape: [batch_size, 81]  — solution tokens (integers 1–9)
```

---

## Why Classification, Not Regression

The model outputs a **probability distribution** over 10 digits for each cell — not a single number. For each cell it produces 10 scores, one per possible digit. The predicted digit is the one with the highest score.

The reason: you need to measure how wrong the model is during training. If the model outputs `3.7` and the answer is `4`, "how wrong is that?" is not a well-defined question for a digit. But if the model outputs `[0.01, 0.01, 0.01, 0.85, 0.05, ...]` and the answer is `4`, you can measure exactly how wrong it is using **cross-entropy loss** — the standard loss function for classification.

Output shape at each step: `[batch_size, 81, 10]` — for each puzzle, for each cell, 10 scores.

---

## The Embedding

The model cannot operate on raw integers — every operation inside a transformer is matrix multiplication, which requires vectors. So each integer token gets converted to a vector of 128 numbers via a **lookup table**.

The embedding table has shape `[10, 128]` — one row per possible digit, each row a vector of 128 numbers:

```
digit 0 → [ 0.2, -0.5,  0.8,  0.1, ...]   # 128 numbers
digit 1 → [ 0.4, -0.8,  0.9,  0.3, ...]
digit 2 → [ 0.3, -0.2,  0.7, -0.1, ...]
...
digit 9 → [ 0.8, -0.2, -0.3,  0.5, ...]
```

When you pass in token `3`, you get back row 3. One vector, 128 numbers. These numbers start random and are adjusted during training until they encode something meaningful about what each digit means in the context of Sudoku.

After embedding, the puzzle goes from shape `[batch, 81]` to `[batch, 81, 128]`.

---

## The Position Embedding

**The problem:** two cells with the same digit get identical embedding vectors. The transformer cannot distinguish A1 from A2 if both contain `0`.

**The fix:** a second lookup table — the position embedding — with shape `[81, 128]`. One row per cell position. Each position has its own vector of 128 numbers.

The two vectors are **added together** to produce the final representation of each cell:

```
A1 (digit 0, position 0):
  embedding(0)     = [ 0.2, -0.5,  0.8,  0.1]
  pos_embedding(0) = [ 0.1,  0.1,  0.1,  0.1]
  final vector     = [ 0.3, -0.4,  0.9,  0.2]

A2 (digit 0, position 1):
  embedding(0)     = [ 0.2, -0.5,  0.8,  0.1]   ← same digit, same row
  pos_embedding(1) = [ 0.2,  0.2,  0.2,  0.2]   ← different position, different row
  final vector     = [ 0.4, -0.3,  1.0,  0.3]   ← different result
```

A1 and A2 now have different vectors even though they contain the same digit. The transformer can distinguish them.

**Important:** the position information never gets decoded or subtracted back out. The transformer operates on the combined vector as-is. Position doesn't need to be recoverable — it just needs to be present so that attention can use it. Both the digit embeddings and the position embeddings are learned jointly during training, adjusted together until the combined vector is useful for prediction.

---

## The Causal Mask

The transformer uses **attention** — each cell can look at other cells and decide what's relevant. But for autoregressive generation, you cannot allow a cell to look at cells to its right. Cell A3 cannot attend to A4 when predicting A3's digit — that information isn't available yet at inference time.

The causal mask enforces this. It is an 81×81 matrix:

```
         A1    A2    A3    A4    A5  ...
A1    [  ok   BLOCK BLOCK BLOCK BLOCK ]
A2    [  ok   ok    BLOCK BLOCK BLOCK ]
A3    [  ok   ok    ok    BLOCK BLOCK ]
A4    [  ok   ok    ok    ok    BLOCK ]
...
```

BLOCK entries are set to `-inf`. When the attention mechanism computes how much cell i should attend to cell j, it adds `-inf` to that score before softmax. `-inf` through softmax becomes 0 — the attention weight is zeroed out. Cell i effectively cannot see cell j.

This is how left-to-right commitment is mechanically enforced.

---

## The Transformer Layers

After embedding + position encoding, the sequence passes through 4 transformer decoder layers. Each layer applies **attention** followed by a **feedforward network**.

Attention asks: for each cell, which previous cells are most relevant to predicting this cell's digit? It computes a weighted average of all visible cells' vectors, where the weights are learned. After attention, each cell's vector has absorbed information from everything to its left.

With 4 heads, the attention mechanism looks for 4 different types of relationships simultaneously. In the Sotaku paper (a similar architecture), different heads learned to attend to rows, columns, and boxes independently — the constraint structure of Sudoku emerged from training, with no rules specified.

After 4 layers, each cell's 128-dimensional vector has been updated 4 times, each time incorporating more context from cells to its left.

---

## The Output Layer

A final linear layer (`nn.Linear(128, 10)`) projects each 128-dimensional vector down to 10 scores — one per possible digit. The shape goes from `[batch, 81, 128]` to `[batch, 81, 10]`.

The predicted digit for each cell is `argmax` over those 10 scores. During training, cross-entropy loss measures how far the full distribution is from the correct answer.

---

## The Full Forward Pass

```
Input: [batch, 81] integers

→ embedding lookup:        [batch, 81, 128]   (digit vectors)
→ + position embedding:    [batch, 81, 128]   (position baked in)
→ transformer (4 layers):  [batch, 81, 128]   (each cell sees leftward context)
→ linear projection:       [batch, 81, 10]    (10 scores per cell)

Output: [batch, 81, 10] scores
```

---

## The Fundamental Weakness

The model commits to each digit left to right, with no ability to revise. When predicting A1, it has seen only the puzzle tokens — no solution tokens at all. For hard puzzles where A1's correct value requires knowing the full board configuration, the model is guessing with minimal information.

This is not a bug in the implementation — it is the defining property of autoregressive generation. The model has to learn to make good guesses from partial information, and for hard constraint satisfaction problems, this is a fundamental architectural limitation.

Concretely: Norvig's solver on a 17-given puzzle (64 ambiguous cells after propagation) takes ~30 seconds of search. The autoregressive model makes a single left-to-right pass in milliseconds — but its accuracy on hard puzzles will be significantly lower than on easy ones, because it has no backtracking and no global constraint reasoning.

This is the gap the diffusion model is designed to close.

---

## Architecture Summary

| Component | Details |
|---|---|
| Vocab size | 10 (digits 0–9) |
| Sequence length | 81 (one token per cell) |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Feedforward dim | 512 |
| Output | Linear(128, 10) |
| Total parameters | ~1,071,242 |
| Training data | 100,000 puzzles (subset of 1M) |
| Epochs | 5 |

---

## What Comes Next

After training this model, it will be benchmarked against Norvig's solver on the same hard puzzles:

- Easy puzzles (~33 givens): Norvig ~2ms, 100% accuracy
- Hard puzzles (~17 givens): Norvig ~30 seconds, 100% accuracy

The autoregressive transformer will be fast on both — a single forward pass. The question is accuracy on hard puzzles. That accuracy gap is the baseline the diffusion model needs to beat.

The diffusion model's advantage: it maintains uncertainty over all cells simultaneously and refines the solution globally, rather than committing left to right. It is architecturally closer to the constraint propagation logic that makes Norvig's solver work — but learned from data rather than hand-coded, making it applicable to problems where the rules are unknown.