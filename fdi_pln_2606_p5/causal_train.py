"""Entrenamiento del LLM causal sobre un corpus de textos."""

import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger


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
    """Los dataloaders se encargan de ir aportando pares para el entrenamiento,
    incluyendo batching, mezcla aleatoria, etc."""
    data = torch.tensor(tokens, dtype=torch.long)

    # Separamos datos en entrenamiento y validación
    split = int(train_ratio * len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    logger.info(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")

    # Los dataloaders implementan utilidades para el entrenamiento de
    # modelos. Devolvemos uno para train y otro para val
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _run_epoch(model, dataloader, label, optimizer=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación.

    Si se pasa optimizer, entrena el modelo (forward + backward + step).
    Si no, evalúa sin calcular gradientes.
    Devuelve la media de loss sobre todos los batches.
    """
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

        # Pase forward, creando el grafo computacional y calculando loss
        _, loss = model(x, y)

        if optimizer:
            # Propaga la pérdida hacia atrás siguiendo el grafo
            loss.backward()
            # Reducimos "gradientes explosivos" para evitar anomalías de train
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Hacemos un paso del optimizador (eg un pequeño paso de descenso
            # siguiendo el gradiente, o lo que determine el optimizador)
            optimizer.step()

        total_loss += loss.item()
        n += 1

        # Progreso cada 10% de los batches
        if n % max(1, total // 10) == 0:
            logger.info(f"{label} | Batch {n}/{total} | Loss={total_loss/n:.4f}")

    # Devolvemos la media de loss en este epoch
    return total_loss / n


def _plot_losses(train_losses, val_losses, path="loss_charts/loss.png"):
    """Grafica la evolución de las pérdidas de entrenamiento y validación."""
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
    logger.info(f"Gráfica guardada en {path}")


def train(model, tokens, epochs, context_size, batch_size, lr, train_ratio):
    """Entrena el modelo de lenguaje causal sobre los tokens dados.

    Realiza `epochs` épocas de entrenamiento con AdamW, registrando train/val
    loss en cada época.
    """
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)

    # El optimizador ajusta los parámetros que le pasamos en función del
    # gradiente (calculado con forward y backward) y la tasa de aprendizaje
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer, label="train")
        val_loss = _run_epoch(model, val_dl, None, label="val  ")

        # Guardamos las pérdidas para graficar al final
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0

        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")
    _plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    import argparse

    from corpus import load_corpus
    from causal_llm import CausalLLM
    from tokenizer import BPETokenizer

    # Hiperparámetros por defecto (se pueden modificar desde línea de comandos)
    parser = argparse.ArgumentParser(description="Entrenar un LLM causal pequeño")
    parser.add_argument("--corpus",     type=str,   default="resources")
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--expansion",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int,   default=300)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=3e-4)
    args = parser.parse_args()

    # Detectamos el dispositivo disponible (GPU si hay, sino CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # GPU Apple Silicon (M1/M2/M3)
    else:
        device = "cpu"
    logger.info(f"Dispositivo: {device}")

    text = load_corpus(args.corpus)

    # Creamos el tokenizador y codificamos el texto en tokens (ids numéricos)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)
    tokens = tokenizer.encode(text)

    logger.info(tokenizer)
    logger.info(f"\nHiperparámetros: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, "
          f"seq_len={args.seq_len}, expansion={args.expansion}, dropout={args.dropout}, "
          f"vocab_size={args.vocab_size}, epochs={args.epochs}, batch_size={args.batch_size}")

    # Creamos el modelo y lo movemos al dispositivo (GPU o CPU)
    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        expansion=args.expansion,
        dropout=args.dropout,
    ).to(device)

    # Contamos el número de parámetros del modelo para tener una idea de su tamaño
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros del modelo: {n_params:,}\n")

    # Entrenamos el modelo con los tokens del corpus
    train(model, tokens, epochs=args.epochs, context_size=args.seq_len, batch_size=args.batch_size, lr=args.lr)

    # Guardamos los pesos del modelo
    SAVE_PATH = "model_weights/model.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    logger.info(f"Pesos guardados en {SAVE_PATH}")

    # Probamos a generar texto a partir de un prompt
    prompt = "alice and the cat were studying for the exam. what "
    pred = model.generate(tokenizer.encode(prompt), max_tokens=200)
    logger.opt(colors=True).info(f"<cyan>{prompt}</cyan>{tokenizer.decode(pred)[:500]}")

    logger.info("--- Fin ---", flush=True)
