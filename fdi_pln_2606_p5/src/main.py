"""CLI del Transformer LLM + NER sobre el corpus de Lewis Carroll.

Uso:
  uv run main.py train-llm            # Entrena el LLM causal
  uv run main.py train-ner            # Fine-tune cabezal NER
  uv run main.py generate "alice..."  # Genera texto a partir de un prompt
  uv run main.py entities texto.txt   # Encuentra entidades en un fichero
"""

import json
import sys
from pathlib import Path

import click
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_corpus
from tokenizer import BPETokenizer
from causal_llm import CausalLLM
from causal_train import train as train_llm_fn
from defaults import DEFAULTS
from ner import NERLLM, NUM_LABELS
from ner_train import load_ner_from_merged, train_ner

console = Console()

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


def _build_tokenizer(corpus: str, vocab_size: int) -> BPETokenizer:
    """Entrena siempre un tokenizador BPE nuevo a partir del corpus."""
    text = load_corpus(corpus)
    return BPETokenizer(text, vocab_size=vocab_size)


def _build_config(
    d_model: int,
    n_heads: int,
    n_layers: int,
    seq_len: int,
    expansion: int,
    dropout: float,
    vocab_size: int,
) -> dict:
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "expansion": expansion,
        "dropout": dropout,
        "vocab_size": vocab_size,
    }


def _build_ner_model(cfg: dict, device: str) -> NERLLM:
    return NERLLM(
        vocab_size=cfg.get("vocab_size", DEFAULTS["vocab_size"]),
        max_seq_len=cfg.get("seq_len", DEFAULTS["seq_len"]),
        d_model=cfg.get("d_model", DEFAULTS["d_model"]),
        n_heads=cfg.get("n_heads", DEFAULTS["n_heads"]),
        n_layers=cfg.get("n_layers", DEFAULTS["n_layers"]),
        expansion=cfg.get("expansion", DEFAULTS["expansion"]),
        dropout=cfg.get("dropout", DEFAULTS["dropout"]),
        num_labels=NUM_LABELS,
    ).to(device)


def _chunk_words(words: list[str], max_words: int = 40) -> list[list[str]]:
    """Divide una lista de palabras en trozos de como máximo max_words."""
    return [words[i : i + max_words] for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Transformer LLM + NER: genera texto e identifica entidades en el corpus de Alice In Wonderland."""


# ── train-llm ───────────────────────────────────────────────────────────────


@cli.command("train-llm")
@click.option(
    "--corpus",
    default="resources",
    show_default=True,
    help="Directorio con ficheros .txt",
)
@click.option(
    "--d-model",
    default=DEFAULTS["d_model"],
    show_default=True,
    help="Dimensión de embeddings",
)
@click.option(
    "--n-heads",
    default=DEFAULTS["n_heads"],
    show_default=True,
    help="Cabezales de atención",
)
@click.option(
    "--n-layers",
    default=DEFAULTS["n_layers"],
    show_default=True,
    help="Capas del transformer",
)
@click.option(
    "--seq-len",
    default=DEFAULTS["seq_len"],
    show_default=True,
    help="Longitud de contexto",
)
@click.option(
    "--vocab-size",
    default=DEFAULTS["vocab_size"],
    show_default=True,
    help="Tamaño del vocabulario BPE",
)
@click.option(
    "--expansion",
    default=DEFAULTS["expansion"],
    show_default=True,
    help="Factor expansión FFN",
)
@click.option(
    "--dropout",
    default=DEFAULTS["dropout"],
    show_default=True,
    help="Tasa de dropout",
)
@click.option(
    "--epochs",
    default=DEFAULTS["epochs"],
    show_default=True,
    help="Épocas de entrenamiento",
)
@click.option(
    "--batch-size",
    default=DEFAULTS["batch_size"],
    show_default=True,
    help="Tamaño de batch",
)
@click.option(
    "--lr",
    default=DEFAULTS["lr"],
    show_default=True,
    help="Learning rate AdamW",
)
@click.option(
    "--model-dir",
    default="model_info",
    show_default=True,
    help="Directorio de salida",
)
def train_llm_cmd(
    corpus,
    d_model,
    n_heads,
    n_layers,
    seq_len,
    vocab_size,
    expansion,
    dropout,
    epochs,
    batch_size,
    lr,
    model_dir,
):
    """Pre-entrena el LLM causal sobre el corpus de texto.

    El tokenizador BPE y los pesos se guardan en --model-dir junto con
    un config.json que documenta los hiperparámetros utilizados.
    """
    device = _device()
    logger.info(f"Dispositivo: {device}")

    text = load_corpus(corpus)
    tokenizer = BPETokenizer(text, vocab_size=vocab_size)
    tokens = tokenizer.encode(text)
    logger.info(
        f"Vocabulario: {tokenizer.vocab_size} tokens | Corpus: {len(tokens):,} tokens"
    )

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=expansion,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros del modelo: {n_params:,}")
    logger.info(
        f"Hiperparámetros: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
        f"seq_len={seq_len}, vocab_size={vocab_size}, expansion={expansion}, "
        f"dropout={dropout}, epochs={epochs}, batch_size={batch_size}, lr={lr}"
    )

    train_llm_fn(
        model, tokens, epochs=epochs, context_size=seq_len, batch_size=batch_size, lr=lr
    )

    # Persistir pesos, tokenizador y config
    Path(model_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(model_dir) / "p5_causal_2606.pth")
    tokenizer.save(Path(model_dir) / "tokenizer.json")
    config = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "vocab_size": tokenizer.vocab_size,
        "expansion": expansion,
        "dropout": dropout,
    }
    json.dump(config, open(Path(model_dir) / "config.json", "w"), indent=2)
    logger.info(f"Modelo guardado en '{model_dir}/'")


# ── train-ner ───────────────────────────────────────────────────────────────


@cli.command("train-ner")
@click.option(
    "--labels",
    default="labels/ner_labels.json",
    show_default=True,
    help="Fichero con las etiquetas NER",
)
@click.option(
    "--epochs",
    default=DEFAULTS["ner_epochs"],
    show_default=True,
    help="Épocas de entrenamiento del NER",
)
@click.option(
    "--lr",
    default=DEFAULTS["ner_lr"],
    show_default=True,
    help="Learning Rate",
)
@click.option(
    "--batch-size",
    default=DEFAULTS["ner_batch"],
    show_default=True,
    help="Tamaño de batch para el entrenamiento del NER",
)
@click.option(
    "--freeze-backbone",
    default=False,
    show_default=True,
    help="Congela el backbone y entrena solo la cabeza NER",
)
@click.option(
    "--weights-path",
    default="model_info/p5_causal_2606.pth",
    show_default=True,
    help="Ruta directa al .pth del LLM preentrenado",
)
@click.option(
    "--corpus",
    default="resources",
    show_default=True,
    help="Directorio con ficheros .txt",
)
@click.option(
    "--d-model",
    default=DEFAULTS["d_model"],
    show_default=True,
    help="Dimensión de embeddings",
)
@click.option(
    "--n-heads",
    default=DEFAULTS["n_heads"],
    show_default=True,
    help="Cabezales de atención",
)
@click.option(
    "--n-layers",
    default=DEFAULTS["n_layers"],
    show_default=True,
    help="Capas del transformer",
)
@click.option(
    "--seq-len",
    default=DEFAULTS["seq_len"],
    show_default=True,
    help="Longitud de contexto",
)
@click.option(
    "--vocab-size",
    default=DEFAULTS["vocab_size"],
    show_default=True,
    help="Tamaño del vocabulario BPE",
)
@click.option(
    "--expansion",
    default=DEFAULTS["expansion"],
    show_default=True,
    help="Factor expansión FFN",
)
@click.option(
    "--dropout",
    default=DEFAULTS["dropout"],
    show_default=True,
    help="Tasa de dropout",
)
@click.option(
    "--model-dir",
    default="model_info",
    show_default=True,
    help="Directorio de salida",
)
def train_ner_cmd(
    labels,
    epochs,
    lr,
    batch_size,
    freeze_backbone,
    weights_path,
    corpus,
    d_model,
    n_heads,
    n_layers,
    seq_len,
    vocab_size,
    expansion,
    dropout,
    model_dir,
):
    """Fine-tune del cabezal NER sobre el corpus etiquetado (merged.json).

    Carga el backbone pre-entrenado del LLM y le añade/entrena una cabeza
    de clasificación por token para las etiquetas o/pi/pc/li/lc.
    """
    device = _device()
    tokenizer = _build_tokenizer(corpus, vocab_size)
    cfg = _build_config(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        expansion=expansion,
        dropout=dropout,
        vocab_size=tokenizer.vocab_size,
    )

    model = _build_ner_model(cfg, device)

    _weights_path = Path(weights_path)
    if _weights_path.exists():
        state = torch.load(_weights_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Backbone cargado de '{_weights_path}' | missing={len(missing)} | unexpected={len(unexpected)}"
        )
    else:
        raise FileNotFoundError(f"Backbone no encontrado en '{_weights_path}'.")

    ner_data = load_ner_from_merged(labels)
    logger.info(f"Datos NER cargados: {len(ner_data)} frases de '{labels}'")

    train_ner(
        model,
        ner_data,
        tokenizer,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
    )

    out_path = Path(model_dir) / "p5_ner_2606.pth"
    torch.save(model.state_dict(), out_path)
    logger.info(f"Pesos NER guardados en '{out_path}'")


# ── generate ────────────────────────────────────────────────────────────────


@cli.command("generate")
@click.argument("prompt")
@click.option("--max-tokens", default=20, show_default=True, help="Tokens a generar")
@click.option(
    "--temperature",
    default=0.8,
    show_default=True,
    help="Temperatura de muestreo (0.1=determinista, 1.5=creativo)",
)
@click.option(
    "--llm-path",
    default="model_info/p5_causal_2606.pth",
    show_default=True,
    help="Ruta directa al fichero .pth del LLM",
)
@click.option(
    "--config-path",
    default=None,
    help="Ruta opcional a config.json con hiperparámetros del modelo)",
)
@click.option(
    "--corpus",
    default="resources",
    show_default=True,
    help="Directorio con ficheros .txt",
)
def generate_cmd(
    prompt,
    max_tokens,
    temperature,
    llm_path,
    config_path,
    corpus,
):
    """Genera texto a partir de PROMPT usando el LLM entrenado."""
    device = _device()
    if config_path:
        _config_path = Path(config_path)
        if not _config_path.exists():
            raise FileNotFoundError(f"Config no encontrada en '{_config_path}'.")
        cfg = json.load(open(_config_path))
    else:
        cfg = DEFAULTS

    tokenizer = _build_tokenizer(corpus, cfg.get("vocab_size", DEFAULTS["vocab_size"]))

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.get("seq_len", DEFAULTS["seq_len"]),
        d_model=cfg.get("d_model", DEFAULTS["d_model"]),
        n_heads=cfg.get("n_heads", DEFAULTS["n_heads"]),
        n_layers=cfg.get("n_layers", DEFAULTS["n_layers"]),
        expansion=cfg.get("expansion", DEFAULTS["expansion"]),
        dropout=0.0,
    ).to(device)

    _llm_path = Path(llm_path)
    if not _llm_path.exists():
        raise FileNotFoundError(f"Pesos LLM no encontrados en '{_llm_path}'.")
    weights = torch.load(_llm_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    prompt_ids = tokenizer.encode(prompt.lower())
    generated_ids = model.generate(
        prompt_ids, max_tokens=max_tokens, temperature=temperature
    )
    generated_txt = tokenizer.decode(generated_ids)

    console.print(
        Panel(
            f"[bold cyan]{prompt}[/bold cyan]{generated_txt}",
            title="Texto generado",
            border_style="green",
        )
    )


# ── entities ────────────────────────────────────────────────────────────────


@cli.command("entities")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--ner-path",
    default="model_info/p5_ner_2606.pth",
    show_default=True,
    help="Ruta directa al fichero .pth del NER",
)
@click.option(
    "--config-path",
    default=None,
    help="Ruta opcional a config.json con hiperparámetros del modelo",
)
@click.option(
    "--corpus",
    default="resources",
    show_default=True,
    help="Directorio con ficheros .txt (para construir el tokenizer)",
)
@click.option(
    "--chunk-words",
    default=40,
    show_default=True,
    help="Palabras por segmento (ajustar si las frases son muy largas)",
)
def entities_cmd(
    file,
    ner_path,
    config_path,
    corpus,
    chunk_words,
):
    """Encuentra entidades nombradas (personas y lugares) en FILE.

    El texto se procesa en segmentos de --chunk-words palabras para respetar
    la longitud máxima de secuencia del modelo.
    """
    device = _device()
    if config_path:
        _config_path = Path(config_path)
        if not _config_path.exists():
            raise FileNotFoundError(f"Config no encontrada en '{_config_path}'.")
        cfg = json.load(open(_config_path))
    else:
        cfg = DEFAULTS

    tokenizer = _build_tokenizer(corpus, cfg.get("vocab_size", DEFAULTS["vocab_size"]))
    cfg = _build_config(
        d_model=cfg.get("d_model", DEFAULTS["d_model"]),
        n_heads=cfg.get("n_heads", DEFAULTS["n_heads"]),
        n_layers=cfg.get("n_layers", DEFAULTS["n_layers"]),
        seq_len=cfg.get("seq_len", DEFAULTS["seq_len"]),
        expansion=cfg.get("expansion", DEFAULTS["expansion"]),
        dropout=cfg.get("dropout", DEFAULTS["dropout"]),
        vocab_size=tokenizer.vocab_size,
    )

    model = _build_ner_model(cfg, device)

    _ner_path = Path(ner_path)
    if not _ner_path.exists():
        raise FileNotFoundError(f"Pesos NER no encontrados en '{_ner_path}'.")
    weights = torch.load(_ner_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    text = Path(file).read_text(encoding="utf-8")
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
    table.add_column("Entidad", style="bold cyan", no_wrap=True)
    table.add_column("Tipo", style="green")
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
