"""Entrenamiento del LLM causal sobre un corpus de textos."""

import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

from corpus import load_corpus
from main import CausalLLM
from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling.

    Cada sample es un par (x, y) de longitud seq_len, donde y es x
    desplazado una posición a la derecha (predecir el siguiente token).
    """

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    data = torch.tensor(tokens, dtype=torch.long)
    split = int(train_ratio * len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    print(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _run_epoch(model, dataloader, optimizer=None):
    """Ejecuta una epoch de entrenamiento (con optimizer) o evaluación (sin él)."""
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if optimizer:
            optimizer.zero_grad()

        _, loss = model(x, y)

        if optimizer:
            loss.backward()
            # Clip para evitar gradientes explosivos
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


def train(model, tokens, epochs=5, context_size=128, batch_size=64, lr=3e-4, train_ratio=0.9):
    """Entrena el modelo de lenguaje causal sobre los tokens dados."""
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        elapsed = time.time() - t0
        print(f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | tiempo={elapsed:.1f}s")

    print(f"Entrenamiento finalizado en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "resources"
    text = load_corpus(corpus_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    VOCAB_SIZE = 300
    CONTEXT_SIZE = 128

    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
    tokens = tokenizer.encode(text)
    print(tokenizer)

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=CONTEXT_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {n_params:,}")

    train(model, tokens, epochs=5, context_size=CONTEXT_SIZE)

    prompt = "alice and the cat were studying for the exam. what "
    pred = model.generate(tokenizer.encode(prompt), max_tokens=200)
    print(f"\n{prompt}{tokenizer.decode(pred)[:500]}")
