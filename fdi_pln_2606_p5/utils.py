"""Funciones de utilidad compartidas para el proyecto."""

from pathlib import Path
from loguru import logger

def load_corpus(path="resources"):
    """Carga y concatena todos los ficheros .txt del directorio dado."""
    files = sorted(Path(path).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No se encontraron .txt en '{path}'")
    return "\n\n".join(p.read_text(encoding="utf-8") for p in files)

def save_losses(train_losses, val_losses, path="logs/loss.txt"):
    """Guarda las pérdidas de entrenamiento y validación en un fichero de texto."""
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch\ttrain_loss\tval_loss\n")
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{i}\t{tr:.6f}\t{va:.6f}\n")
    logger.info(f"Pérdidas guardadas en {path}")