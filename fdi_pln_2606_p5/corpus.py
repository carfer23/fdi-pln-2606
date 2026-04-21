"""Utilidad para cargar un corpus de ficheros .txt."""

from pathlib import Path


def load_corpus(path="resources"):
    """Carga y concatena todos los ficheros .txt del directorio dado."""
    files = sorted(Path(path).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No se encontraron .txt en '{path}'")
    return "\n\n".join(p.read_text(encoding="utf-8") for p in files)
