"""Fine-tuning del modelo NER sobre el corpus etiquetado de Lewis Carroll."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from ner import NERLLM, NUM_LABELS, load_ner_from_merged, train_ner


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    import argparse

    from tokenizer import BPETokenizer

    parser = argparse.ArgumentParser(description="Fine-tune del cabezal NER")
    parser.add_argument("--data",            type=str,   default="merged.json",   help="Fichero merged.json etiquetado")
    parser.add_argument("--weights-dir",     type=str,   default="model_weights", help="Directorio con pesos LLM y tokenizador")
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Congela el backbone y entrena solo la cabeza NER")
    args = parser.parse_args()

    device = _device()
    logger.info(f"Dispositivo: {device}")

    # Cargar tokenizador y config del LLM pre-entrenado
    tok_path = Path(args.weights_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizador no encontrado en '{tok_path}'.\n"
            "Ejecuta primero: python causal_train.py"
        )
    tokenizer = BPETokenizer.load(tok_path)

    cfg_path = Path(args.weights_dir) / "config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}

    # Construir modelo NER con la misma arquitectura que el LLM
    model = NERLLM(
        vocab_size  = cfg.get("vocab_size",  300),
        max_seq_len = cfg.get("seq_len",     128),
        d_model     = cfg.get("d_model",     128),
        n_heads     = cfg.get("n_heads",     4),
        n_layers    = cfg.get("n_layers",    4),
        expansion   = cfg.get("expansion",   4),
        dropout     = cfg.get("dropout",     0.1),
        num_labels  = NUM_LABELS,
    ).to(device)

    # Cargar pesos del backbone (strict=False ignora lm_head, que no existe en NERLLM)
    llm_path = Path(args.weights_dir) / "model.pth"
    if llm_path.exists():
        state = torch.load(llm_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Backbone cargado de '{llm_path}' | missing={missing} | unexpected={unexpected}")
    else:
        logger.warning(f"No se encontró {llm_path}; entrenando NER desde cero")

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("ner_head"):
                param.requires_grad_(False)
        logger.info("Backbone congelado — solo se entrena la cabeza NER")

    n_params       = sum(p.numel() for p in model.parameters())
    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros totales: {n_params:,} | entrenables: {n_params_train:,}")
    logger.info(
        f"Hiperparámetros NER: epochs={args.epochs}, lr={args.lr}, "
        f"batch_size={args.batch_size}, freeze_backbone={args.freeze_backbone}"
    )

    # Cargar datos etiquetados
    ner_data = load_ner_from_merged(args.data, tokenizer)
    logger.info(f"Datos NER cargados: {len(ner_data)} frases de '{args.data}'")

    # Fine-tuning
    train_ner(model, ner_data, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    # Guardar pesos NER
    out_path = Path(args.weights_dir) / "ner_model.pth"
    torch.save(model.state_dict(), out_path)
    logger.info(f"Pesos NER guardados en '{out_path}'")
