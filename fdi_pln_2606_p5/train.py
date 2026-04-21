"""Entrenamiento del LLM causal sobre un corpus de textos."""

import time

import matplotlib.pyplot as plt
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


def _run_epoch(model, dataloader, optimizer=None, label=""):
    """Ejecuta una epoch de entrenamiento (con optimizer) o evaluación (sin él)."""
    total_loss, n = 0, 0
    total = len(dataloader)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1
        # Progreso cada 10% de los batches
        if n % max(1, total // 10) == 0:
            print(f"  {label} batch {n}/{total} | loss={total_loss/n:.4f}", end="\r")

    print()  # salto de línea al terminar
    return total_loss / n


def _plot_losses(train_losses, val_losses, path="loss.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses,   marker="o", label="Val loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Gráfica guardada en {path}")


def train(model, tokens, epochs=10, context_size=128, batch_size=64, lr=3e-4, train_ratio=0.9):
    """Entrena el modelo de lenguaje causal sobre los tokens dados."""
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(epochs):
        print(f"\nEpoca {epoch + 1}/{epochs}")
        train_loss = _run_epoch(model, train_dl, optimizer, label="train")
        val_loss = _run_epoch(model, val_dl, None, label="val  ")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - t0
        print(f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | tiempo={elapsed:.1f}s")

    print(f"Entrenamiento finalizado en {time.time() - t0:.1f}s")
    _plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    import argparse

    # Hiperparámetros por defecto — edita aquí para cambiarlos sin pasar argumentos
    D_MODEL    = 128
    N_HEADS    = 4
    N_LAYERS   = 4
    SEQ_LEN    = 128
    EXPANSION  = 4
    DROPOUT    = 0.1
    VOCAB_SIZE = 300
    EPOCHS     = 5
    BATCH_SIZE = 40
    LR         = 3e-4

    parser = argparse.ArgumentParser(description="Entrenar un LLM causal pequeño")
    parser.add_argument("corpus", nargs="?", default="resources", help="Directorio con .txt")
    parser.add_argument("--d_model",    type=int,   default=D_MODEL)
    parser.add_argument("--n_heads",    type=int,   default=N_HEADS)
    parser.add_argument("--n_layers",   type=int,   default=N_LAYERS)
    parser.add_argument("--seq_len",    type=int,   default=SEQ_LEN)
    parser.add_argument("--expansion",  type=int,   default=EXPANSION)
    parser.add_argument("--dropout",    type=float, default=DROPOUT)
    parser.add_argument("--vocab_size", type=int,   default=VOCAB_SIZE)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # GPU Apple Silicon (M1/M2/M3)
    else:
        device = "cpu"
    print(f"Dispositivo: {device}")

    text = load_corpus(args.corpus)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    tokens = tokenizer.encode(text)
    print(tokenizer)
    print(f"\nHiperparámetros: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, "
          f"seq_len={args.seq_len}, expansion={args.expansion}, dropout={args.dropout}, "
          f"vocab_size={args.vocab_size}, epochs={args.epochs}, batch_size={args.batch_size}")

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        expansion=args.expansion,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {n_params:,}\n")

    train(model, tokens, epochs=args.epochs, context_size=args.seq_len, batch_size=args.batch_size, lr=args.lr)

    torch.save(model.state_dict(), "model.pth")
    print("Pesos guardados en model.pth")

    # Generamos en CPU para evitar problemas con MPS/multinomial
    model.to("cpu")
    prompt = "alice and the cat were studying for the exam. what "
    print("\n--- Texto generado ---")
    try:
        pred = model.generate(tokenizer.encode(prompt), max_tokens=200)
        print(prompt + tokenizer.decode(pred)[:500])
    except Exception as e:
        print(f"Error al generar: {e}")
    print("--- Fin ---", flush=True)
