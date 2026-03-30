import torch
import pandas as pd

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

X, Y = load_dataset('sudoku.csv', n=10000)


import torch.nn as nn

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

model = SudokuTransformer()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

