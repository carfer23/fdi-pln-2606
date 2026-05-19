#!/usr/bin/env python3
"""CLI del Transformer LLM + NER sobre el corpus de Lewis Carroll.

Uso:
  python main.py train-llm            # Entrena el LLM causal
  python main.py train-ner            # Fine-tune cabezal NER
  python main.py generate "alice..."  # Genera texto a partir de un prompt
  python main.py entities texto.txt   # Encuentra entidades en un fichero

"""

import json
import re
import sys
from pathlib import Path

import click
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Añadimos el directorio al path para importar los módulos vecinos
sys.path.insert(0, str(Path(__file__).parent))

from corpus import load_corpus
from tokenizer import BPETokenizer
from causal_llm import CausalLLM
from causal_train import train as train_llm_fn
from ner import NERLLM, LABEL2ID, NUM_LABELS, load_ner_from_merged, train_ner

console = Console()

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto 
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Arquitectura del backbone compartido
    "d_model":    128,   # Dimensión de embeddings y representaciones internas
    "n_heads":    4,     # Cabezales de atención; head_dim = d_model / n_heads = 32
    "n_layers":   4,     # Bloques transformer apilados
    "seq_len":    128,   # Longitud máxima de contexto / secuencia
    "expansion":  4,     # Factor de expansión de la capa feed-forward (hidden = 512)
    "dropout":    0.1,   # Regularización; desactivado automáticamente en eval()
    "vocab_size": 300,   # Tokens BPE; suficiente para el vocabulario de 
    # Entrenamiento LLM
    "epochs":     5,
    "batch_size": 32,
    "lr":         3e-4,
    # Fine-tuning NER
    "ner_epochs":    15,
    "ner_lr":        1e-4,  # LR más bajo para preservar representaciones pre-entrenadas
    "ner_batch":     8,
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _device() -> str:
    """Detecta el dispositivo de cómputo disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def _load_tokenizer(weights_dir: str) -> BPETokenizer:
    """Carga el tokenizador guardado; lanza error descriptivo si no existe."""
    path = Path(weights_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizador no encontrado en '{path}'.\n"
            "Ejecuta primero: python main.py train-llm"
        )
    return BPETokenizer.load(path)


def _load_config(weights_dir: str) -> dict:
    """Carga la configuración del modelo; usa DEFAULTS si no existe."""
    path = Path(weights_dir) / "config.json"
    if path.exists():
        return json.load(open(path))
    logger.warning("config.json no encontrado, usando hiperparámetros por defecto")
    return DEFAULTS.copy()


def _build_ner_model(cfg: dict, device: str) -> NERLLM:
    return NERLLM(
        vocab_size  = cfg.get("vocab_size",  DEFAULTS["vocab_size"]),
        max_seq_len = cfg.get("seq_len",     DEFAULTS["seq_len"]),
        d_model     = cfg.get("d_model",     DEFAULTS["d_model"]),
        n_heads     = cfg.get("n_heads",     DEFAULTS["n_heads"]),
        n_layers    = cfg.get("n_layers",    DEFAULTS["n_layers"]),
        expansion   = cfg.get("expansion",   DEFAULTS["expansion"]),
        dropout     = cfg.get("dropout",     DEFAULTS["dropout"]),
        num_labels  = NUM_LABELS,
    ).to(device)


def _chunk_words(words: list[str], max_words: int = 40) -> list[list[str]]:
    """Divide una lista de palabras en trozos de como máximo max_words."""
    return [words[i : i + max_words] for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Transformer LLM + NER: genera texto e identifica entidades en Carroll."""


# ── train-llm ───────────────────────────────────────────────────────────────

@cli.command("train-llm")
@click.option("--corpus",      default="resources",           show_default=True, help="Directorio con ficheros .txt")
@click.option("--d-model",     default=DEFAULTS["d_model"],   show_default=True, help="Dimensión de embeddings")
@click.option("--n-heads",     default=DEFAULTS["n_heads"],   show_default=True, help="Cabezales de atención")
@click.option("--n-layers",    default=DEFAULTS["n_layers"],  show_default=True, help="Capas del transformer")
@click.option("--seq-len",     default=DEFAULTS["seq_len"],   show_default=True, help="Longitud de contexto")
@click.option("--vocab-size",  default=DEFAULTS["vocab_size"],show_default=True, help="Tamaño del vocabulario BPE")
@click.option("--expansion",   default=DEFAULTS["expansion"], show_default=True, help="Factor expansión FFN")
@click.option("--dropout",     default=DEFAULTS["dropout"],   show_default=True, help="Tasa de dropout")
@click.option("--epochs",      default=DEFAULTS["epochs"],    show_default=True, help="Épocas de entrenamiento")
@click.option("--batch-size",  default=DEFAULTS["batch_size"],show_default=True, help="Tamaño de batch")
@click.option("--lr",          default=DEFAULTS["lr"],        show_default=True, help="Learning rate AdamW")
@click.option("--weights-dir", default="model_weights",       show_default=True, help="Directorio de salida")
def train_llm_cmd(corpus, d_model, n_heads, n_layers, seq_len, vocab_size,
                  expansion, dropout, epochs, batch_size, lr, weights_dir):
    """Entrena el LLM causal sobre el corpus de texto.

    El tokenizador BPE y los pesos se guardan en --weights-dir junto con
    un config.json que documenta los hiperparámetros usados.
    """
    device = _device()
    logger.info(f"Dispositivo: {device}")

    text      = load_corpus(corpus)
    tokenizer = BPETokenizer(text, vocab_size=vocab_size)
    tokens    = tokenizer.encode(text)
    logger.info(f"Vocabulario: {tokenizer.vocab_size} tokens | Corpus: {len(tokens):,} tokens")

    model = CausalLLM(
        vocab_size  = tokenizer.vocab_size,
        max_seq_len = seq_len,
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = n_layers,
        expansion   = expansion,
        dropout     = dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros del modelo: {n_params:,}")
    logger.info(
        f"Hiperparámetros: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
        f"seq_len={seq_len}, vocab_size={vocab_size}, expansion={expansion}, "
        f"dropout={dropout}, epochs={epochs}, batch_size={batch_size}, lr={lr}"
    )

    train_llm_fn(model, tokens, epochs=epochs, context_size=seq_len,
                 batch_size=batch_size, lr=lr)

    # Persistir pesos, tokenizador y config
    Path(weights_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(weights_dir) / "model.pth")
    tokenizer.save(Path(weights_dir) / "tokenizer.json")
    config = {
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
        "seq_len": seq_len,  "vocab_size": tokenizer.vocab_size,
        "expansion": expansion, "dropout": dropout,
    }
    json.dump(config, open(Path(weights_dir) / "config.json", "w"), indent=2)
    logger.info(f"Modelo guardado en '{weights_dir}/'")


# ── train-ner ───────────────────────────────────────────────────────────────

@cli.command("train-ner")
@click.option("--data",         default="merged.json",   show_default=True, help="Fichero merged.json etiquetado")
@click.option("--weights-dir",  default="model_weights", show_default=True, help="Directorio con pesos LLM y tokenizador")
@click.option("--epochs",       default=DEFAULTS["ner_epochs"], show_default=True)
@click.option("--lr",           default=DEFAULTS["ner_lr"],     show_default=True, help="LR (recomendado < lr_LLM)")
@click.option("--batch-size",   default=DEFAULTS["ner_batch"],  show_default=True)
@click.option("--freeze-backbone/--no-freeze-backbone", default=False,
              help="Congela el backbone y entrena solo la cabeza NER")
def train_ner_cmd(data, weights_dir, epochs, lr, batch_size, freeze_backbone):
    """Fine-tune del cabezal NER sobre el corpus etiquetado (merged.json).

    Carga el backbone pre-entrenado del LLM y le añade/entrena una cabeza
    de clasificación por token para las etiquetas o/pi/pc/li/lc.
    """
    device    = _device()
    tokenizer = _load_tokenizer(weights_dir)
    cfg       = _load_config(weights_dir)

    model = _build_ner_model(cfg, device)

    # Cargar pesos del backbone pre-entrenado (strict=False ignora lm_head)
    llm_path = Path(weights_dir) / "model.pth"
    if llm_path.exists():
        state = torch.load(llm_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Backbone cargado de '{llm_path}' | missing={missing} | unexpected={unexpected}")
    else:
        logger.warning(f"No se encontró {llm_path}; entrenando NER desde cero")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("ner_head"):
                param.requires_grad_(False)
        logger.info("Backbone congelado — solo se entrena la cabeza NER")

    ner_data = load_ner_from_merged(data, tokenizer)
    logger.info(f"Datos NER cargados: {len(ner_data)} frases de '{data}'")

    train_ner(model, ner_data, epochs=epochs, lr=lr, batch_size=batch_size)

    out_path = Path(weights_dir) / "ner_model.pth"
    torch.save(model.state_dict(), out_path)
    logger.info(f"Pesos NER guardados en '{out_path}'")


# ── generate ────────────────────────────────────────────────────────────────

@cli.command("generate")
@click.argument("prompt")
@click.option("--max-tokens",  default=200,  show_default=True, help="Tokens a generar")
@click.option("--temperature", default=0.8,  show_default=True, help="Temperatura de muestreo (0.1=determinista, 1.5=creativo)")
@click.option("--weights-dir", default="model_weights", show_default=True)
def generate_cmd(prompt, max_tokens, temperature, weights_dir):
    """Genera texto a partir de PROMPT usando el LLM entrenado."""
    device    = _device()
    tokenizer = _load_tokenizer(weights_dir)
    cfg       = _load_config(weights_dir)

    model = CausalLLM(
        vocab_size  = cfg.get("vocab_size",  DEFAULTS["vocab_size"]),
        max_seq_len = cfg.get("seq_len",     DEFAULTS["seq_len"]),
        d_model     = cfg.get("d_model",     DEFAULTS["d_model"]),
        n_heads     = cfg.get("n_heads",     DEFAULTS["n_heads"]),
        n_layers    = cfg.get("n_layers",    DEFAULTS["n_layers"]),
        expansion   = cfg.get("expansion",   DEFAULTS["expansion"]),
        dropout     = 0.0,  # Sin dropout en inferencia
    ).to(device)

    weights = torch.load(Path(weights_dir) / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(weights)

    prompt_ids    = tokenizer.encode(prompt.lower())
    generated_ids = model.generate(prompt_ids, max_tokens=max_tokens, temperature=temperature)
    generated_txt = tokenizer.decode(generated_ids)

    console.print(Panel(
        f"[bold cyan]{prompt}[/bold cyan]{generated_txt}",
        title="Texto generado",
        border_style="green",
    ))


# ── entities ────────────────────────────────────────────────────────────────

@cli.command("entities")
@click.argument("file", type=click.Path(exists=True))
@click.option("--weights-dir", default="model_weights", show_default=True)
@click.option("--chunk-words", default=40, show_default=True,
              help="Palabras por segmento (ajustar si las frases son muy largas)")
def entities_cmd(file, weights_dir, chunk_words):
    """Encuentra entidades nombradas (personas y lugares) en FILE.

    El texto se procesa en segmentos de --chunk-words palabras para respetar
    la longitud máxima de secuencia del modelo.
    """
    device    = _device()
    tokenizer = _load_tokenizer(weights_dir)
    cfg       = _load_config(weights_dir)

    model = _build_ner_model(cfg, device)

    ner_path = Path(weights_dir) / "ner_model.pth"
    if not ner_path.exists():
        raise FileNotFoundError(
            f"Pesos NER no encontrados en '{ner_path}'.\n"
            "Ejecuta primero: python main.py train-ner"
        )
    weights = torch.load(ner_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    text  = Path(file).read_text(encoding="utf-8")
    # Normalizamos a minúsculas (el corpus Carroll está en minúsculas)
    words = text.lower().split()

    all_entities: list[tuple[str, str]] = []
    for chunk in _chunk_words(words, chunk_words):
        all_entities.extend(model.predict_entities(chunk, tokenizer))

    # Deduplicar preservando orden de primera aparición
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for ent, kind in all_entities:
        key = (ent.lower(), kind)
        if key not in seen:
            seen.add(key)
            unique.append((ent, kind))

    if not unique:
        console.print("[yellow]No se encontraron entidades nombradas.[/yellow]")
        return

    table = Table(title=f"Entidades en '{Path(file).name}'", show_lines=True)
    table.add_column("Entidad",    style="bold cyan", no_wrap=True)
    table.add_column("Tipo",       style="green")
    table.add_column("Apariciones", justify="right")

    # Contar ocurrencias en la lista completa (antes de deduplicar)
    from collections import Counter
    counts = Counter((e.lower(), k) for e, k in all_entities)

    for ent, kind in unique:
        n = counts[(ent.lower(), kind)]
        table.add_row(ent, kind, str(n))

    console.print(table)


if __name__ == "__main__":
    cli()
