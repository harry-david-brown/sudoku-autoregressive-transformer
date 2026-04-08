import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(path, n=None):
    df = pd.read_csv(path, dtype=str)
    puzzles = df['quizzes'].tolist()
    solutions = df['solutions'].tolist()
    if n:
        puzzles = puzzles[:n]
        solutions = solutions[:n]
    X = torch.tensor([[int(c) for c in p] for p in puzzles], dtype=torch.long)
    Y = torch.tensor([[int(c) for c in s] for s in solutions], dtype=torch.long)
    return X, Y


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


# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

puzzles, solutions = load_dataset('sudoku.csv', n=500000)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SudokuTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training ───────────────────────────────────────────────────────────────────
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start = time.time()
    for batch_puzzles, batch_solutions in loader:
        batch_puzzles = batch_puzzles.to(device)
        batch_solutions = batch_solutions.to(device)
        optimizer.zero_grad()
        output = model(batch_puzzles)
        loss = criterion(output.permute(0, 2, 1), batch_solutions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {total_loss/len(loader):.4f} — {elapsed:.0f}s")

torch.save(model.state_dict(), 'sudoku_transformer.pth')
print("Model saved.")

# ── Evaluation ─────────────────────────────────────────────────────────────────
model.eval()
correct_cells = 0
correct_puzzles = 0
total_cells = 0
total_puzzles = 0
position_correct = torch.zeros(81)
position_total = torch.zeros(81)

failed_puzzles = []

with torch.no_grad():
    for batch_puzzles, batch_solutions in loader:
        batch_puzzles = batch_puzzles.to(device)
        batch_solutions = batch_solutions.to(device)
        output = model(batch_puzzles)
        predictions = output.argmax(dim=-1)
        cell_correct = (predictions == batch_solutions)

        correct_cells += cell_correct.sum().item()
        total_cells += batch_solutions.numel()

        position_correct += cell_correct.cpu().sum(dim=0).float()
        position_total += batch_solutions.shape[0]

        puzzle_correct = cell_correct.all(dim=-1)
        correct_puzzles += puzzle_correct.sum().item()
        total_puzzles += batch_solutions.shape[0]

        # collect failed puzzles for analysis
        if len(failed_puzzles) < 10:
            for i in range(batch_puzzles.shape[0]):
                if not puzzle_correct[i] and len(failed_puzzles) < 10:
                    failed_puzzles.append({
                        'puzzle':     batch_puzzles[i].cpu(),
                        'prediction': predictions[i].cpu(),
                        'solution':   batch_solutions[i].cpu(),
                    })

print(f"\nCell accuracy:   {correct_cells/total_cells*100:.2f}%")
print(f"Puzzle accuracy: {correct_puzzles/total_puzzles*100:.2f}%")

# ── Position analysis ──────────────────────────────────────────────────────────
position_acc = (position_correct / position_total * 100).tolist()
print("\nAccuracy by board position (row by row):")
for row in range(9):
    accs = [f"{position_acc[row*9+col]:5.1f}" for col in range(9)]
    print(" | ".join(accs))

# ── Failed puzzle analysis ─────────────────────────────────────────────────────
def display_comparison(puzzle, prediction, solution):
    rows = 'ABCDEFGHI'
    print(f"{'PUZZLE':^27}   {'PREDICTION':^27}   {'SOLUTION':^27}")
    for r in range(9):
        p_row, pred_row, sol_row = [], [], []
        for c in range(9):
            idx = r * 9 + c
            p_row.append(str(puzzle[idx].item()))
            pred_row.append(str(prediction[idx].item()))
            sol_row.append(str(solution[idx].item()))
        sep = lambda row, c: " | " if (c+1) % 3 == 0 and c != 8 else " "
        fmt = lambda row: "".join(row[i] + sep(row, i) for i in range(9))
        print(f"{fmt(p_row):27}   {fmt(pred_row):27}   {fmt(sol_row):27}")
        if r in [2, 5]:
            print("-" * 89)

print("\n── Failed puzzle examples ──")
for i, ex in enumerate(failed_puzzles[:3]):
    print(f"\nExample {i+1}:")
    display_comparison(ex['puzzle'], ex['prediction'], ex['solution'])