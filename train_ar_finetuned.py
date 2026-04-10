import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader

HARD_DATASET_PATH = '/Users/harry/.cache/huggingface/hub/datasets--imone--sudoku-hard-v2/snapshots/58942f96baeb572ca3127e2a9e9c70f330783d6b/train.csv'


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

puzzles, solutions = load_hard_dataset(HARD_DATASET_PATH, n=500000)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SudokuTransformer().to(device)
model.load_state_dict(torch.load('sudoku_transformer.pth', map_location=device))
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

torch.save(model.state_dict(), 'sudoku_transformer_finetuned_hard.pth')
print("Model saved.")
