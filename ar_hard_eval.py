import torch
import torch.nn as nn
import pandas as pd
import time
import random
from torch.utils.data import TensorDataset, DataLoader

HARD_DATASET_PATH = 'train.csv'

def load_hard_dataset(path, n=None, min_rating=50):
    df = pd.read_csv(path, dtype=str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > min_rating].copy()
    print(f"Hard puzzles available (rating > {min_rating}): {len(df):,}")
    if n:
        df = df.iloc[:n]
    puzzles   = df['question'].tolist()
    solutions = df['answer'].tolist()
    X = torch.tensor([[0 if c == '.' else int(c) for c in p] for p in puzzles], dtype=torch.long)
    Y = torch.tensor([[0 if c == '.' else int(c) for c in s] for s in solutions], dtype=torch.long)
    return X, Y, df['rating'].iloc[:n if n else len(df)].tolist()


class SudokuTransformer(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=128, num_heads=4, num_layers=4, seq_len=81):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len, device=x.device)
        x = self.transformer(x, x, tgt_mask=mask)
        return self.output(x)


def check_validity(grid):
    digits = set('123456789')
    violations = []
    for r in range(9):
        if set(grid[r*9:(r+1)*9]) != digits: violations.append(f'Row {r+1}')
    for c in range(9):
        if set(grid[c::9]) != digits: violations.append(f'Col {c+1}')
    for br in range(3):
        for bc in range(3):
            box = [grid[(br*3+r)*9+(bc*3+c)] for r in range(3) for c in range(3)]
            if set(box) != digits: violations.append(f'Box ({br+1},{bc+1})')
    return violations


# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

puzzles, solutions, ratings = load_hard_dataset(HARD_DATASET_PATH, n=10000)
print(f"Loaded {len(puzzles):,} hard puzzles")
print(f"Rating range: {min(ratings):.0f} – {max(ratings):.0f}, mean {sum(ratings)/len(ratings):.1f}")

model = SudokuTransformer().to(device)
model.load_state_dict(torch.load('sudoku_transformer.pth', map_location=device))
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ── Evaluation — one-shot ──────────────────────────────────────────────────────
print("\n=== AR Transformer — Hard Puzzles (one-shot) ===")

sample_size = 1000
indices = random.sample(range(len(puzzles)), sample_size)
sample_puzzles   = [puzzles[i] for i in indices]
sample_solutions = [solutions[i] for i in indices]
sample_ratings   = [ratings[i] for i in indices]

correct_cells   = 0
correct_puzzles = 0
total_cells     = 0
total_violations = 0
puzzles_with_violations = 0
position_correct = torch.zeros(81)
position_total   = torch.zeros(81)

with torch.no_grad():
    for puz, sol in zip(sample_puzzles, sample_solutions):
        x = puz.unsqueeze(0).to(device)
        output = model(x)
        pred = output.argmax(dim=-1)[0]

        unknown = (puz == 0)
        cell_correct = (pred.cpu() == sol) & unknown
        correct_cells += cell_correct.sum().item()
        total_cells   += unknown.sum().item()
        position_correct += cell_correct.float()
        position_total   += unknown.float()

        # keep givens in prediction
        pred_str = ''
        for i in range(81):
            if puz[i].item() != 0:
                pred_str += str(puz[i].item())
            else:
                pred_str += str(pred[i].item())

        v = check_validity(pred_str)
        total_violations += len(v)
        if v: puzzles_with_violations += 1
        if pred_str == ''.join(str(s.item()) for s in sol):
            correct_puzzles += 1

print(f"Cell accuracy:         {correct_cells/total_cells*100:.2f}%")
print(f"Puzzle accuracy:       {correct_puzzles/sample_size*100:.2f}%")
print(f"Puzzles w/ violations: {puzzles_with_violations}/{sample_size}")
print(f"Avg violations:        {total_violations/sample_size:.2f}")

print("\nPer-position accuracy (unknown cells only):")
position_acc = (position_correct / position_total.clamp(min=1) * 100).tolist()
for row in range(9):
    print(' | '.join(f'{position_acc[row*9+col]:5.1f}' for col in range(9)))

# ── Difficulty breakdown ───────────────────────────────────────────────────────
print("\n=== Accuracy by difficulty tier ===")
tiers = [(50, 100), (100, 200), (200, 465)]
for low, high in tiers:
    tier_indices = [i for i, r in enumerate(sample_ratings) if low < r <= high]
    if not tier_indices:
        print(f"  Rating {low}-{high}: no samples")
        continue
    tier_correct = 0
    tier_total   = 0
    tier_violations = 0
    with torch.no_grad():
        for i in tier_indices:
            puz = sample_puzzles[i]
            sol = sample_solutions[i]
            x = puz.unsqueeze(0).to(device)
            output = model(x)
            pred = output.argmax(dim=-1)[0]
            unknown = (puz == 0)
            tier_correct += ((pred.cpu() == sol) & unknown).sum().item()
            tier_total   += unknown.sum().item()
            pred_str = ''.join(
                str(puz[i].item()) if puz[i].item() != 0 else str(pred[i].item())
                for i in range(81)
            )
            tier_violations += len(check_validity(pred_str))
    print(f"  Rating {low}-{high} (n={len(tier_indices)}): "
          f"Cell acc {tier_correct/tier_total*100:.1f}%, "
          f"Avg violations {tier_violations/len(tier_indices):.1f}")
