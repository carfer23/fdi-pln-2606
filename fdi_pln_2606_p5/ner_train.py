"""Entrenamiento (fine-tuning) del modelo NER sobre un CausalLLM preentrenado."""

import json
import time
import pathlib
import sys
import torch
from torch.utils.data import DataLoader
from loguru import logger

from ner import NERLLM, NERDataset, collate_ner, NUM_LABELS
from tokenizer import BPETokenizer

def load_ner_from_merged(path="labels/merged.json"):
    """
    Lee el archivo merged.json que contiene la lista de diccionarios.
    Cada diccionario debe tener "tokens" (o "words") y "labels".
    Devuelve una lista de tuplas (lista_de_palabras, lista_de_etiquetas_BIO_o_similar).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # En el json: "tokens" y "labels".
        # Palabras y etiquetas alineadas a nivel de palabra
        words = item.get("tokens", [])
        labels = item.get("labels", [])
        
        if len(words) > 0 and len(words) == len(labels):
            samples.append((words, labels))
    
    return samples

def _make_dataloaders(ner_data, tokenizer, batch_size, train_ratio=0.9):
    """Crea DataLoaders para entrenamiento y validación con padding dinámico."""
    split = int(train_ratio * len(ner_data))
    train_ds = NERDataset(ner_data[:split], tokenizer)
    val_ds = NERDataset(ner_data[split:], tokenizer)
    logger.info(f"Train: {len(train_ds):,} frases, Val: {len(val_ds):,}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ner),
        DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_ner),
    )

def _run_epoch(model, dataloader, label, optimizer=None, class_weights=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación."""
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

        _, loss = model(x, y, class_weights=class_weights)

        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

        if n % max(1, total // 10) == 0:
            logger.info(f"{label} | Batch {n}/{total} | Loss={total_loss/n:.4f}")

    return total_loss / n

def _compute_class_weights(train_ds):
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase en el dataset."""
    counts = torch.zeros(NUM_LABELS, dtype=torch.long)
    for _, labels in train_ds.samples:
        valid = labels[labels >= 0]
        if valid.numel() == 0:
            continue
        counts += torch.bincount(valid, minlength=NUM_LABELS)
    counts = counts.clamp_min(1)
    weights = counts.sum().float() / counts.float()
    weights = weights / weights.mean()
    return weights


def train_ner(model, ner_data, tokenizer, epochs, batch_size, lr):
    train_dl, val_dl = _make_dataloaders(ner_data, tokenizer, batch_size)

    # Congelar el backbone (Transformer base)
    for name, param in model.named_parameters():
        if "ner_head" not in name:
            param.requires_grad = False
    
    # Podemos aplicar un learning rate diferente para el backbone y para la nueva cabeza lineal
    # Asignamos todo a AdamW por simplicidad, para fine-tuning se recomienda un lr bajo
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    class_weights = _compute_class_weights(train_dl.dataset)
    class_weights = class_weights.to(next(model.parameters()).device)
    logger.info(f"Pesos por clase: {class_weights.tolist()}")

    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(
            model,
            train_dl,
            "train",
            optimizer=optimizer,
            class_weights=class_weights,
        )
        val_loss = _run_epoch(
            model,
            val_dl,
            "val",
            optimizer=None,
            class_weights=class_weights,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )
    
    # Importar save_losses desde utils
    from utils import save_losses
    save_losses(train_losses, val_losses, path="logs/ner_loss.txt")
    logger.info(f"Fine-tuning finalizado en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    import re
    from utils import load_corpus
    
    parser = argparse.ArgumentParser(description="Fine-tuning para NER de un CausalLLM preentrenado")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-5) # Menor lr para fine-tuning
    parser.add_argument("--merged_json", type=str,  default="labels/merged.json")
    parser.add_argument("--model_path", type=str,   default="model_info/p5_causal_2606.pth")
    parser.add_argument("--corpus",     type=str,   default="resources")
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=3)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--expansion",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int,   default=300)
    args = parser.parse_args()

    # Almacenar logs
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.add(logs_dir / "ner.log", rotation="50 MB")

    # 1. Detectar dispositivo
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Usando {device} para fine-tuning NER.")

    # 2. Entrenar tokenizador
    logger.info("Entrenando tokenizador BPE...")
    text = load_corpus(args.corpus)
    tokenizer = BPETokenizer(text, vocab_size=args.vocab_size)

    # 3. Instanciar el modelo NER
    model = NERLLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        expansion=args.expansion,
        dropout=args.dropout,
        num_labels=NUM_LABELS
    )

    # 4. Cargar los pesos preentrenados en el backbone
    pth_path = pathlib.Path(args.model_path)
    if not pth_path.exists():
        logger.error(f"Falta el modelo preentrenado en {pth_path}")
        sys.exit(1)
        
    logger.info("Cargando pesos preentrenados del backbone...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    # Al usar strict=False, cargará las capas comunes de Transformer y omitirá el aviso 
    # de que le faltan los pesos de ner_head (porque los acaba de crear aleatorios)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)

    # 5. Cargar datos de etiquetas para el Fine-Tuning
    logger.info(f"Cargando dataset etiquetado desde {args.merged_json}...")
    ner_data = load_ner_from_merged(args.merged_json)
    
    if not ner_data:
        logger.error("No se encontraron ejemplos válidos en el archivo JSON.")
        sys.exit(1)

    # 6. Ejecutar bucle de entrenamiento (Fine-Tuning)
    train_ner(model, ner_data, tokenizer, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # 7. Guardar modelo NER
    model_dir = pathlib.Path(args.model_path).parent
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "p5_ner_2606.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Modelo NER guardado en {save_path}")

    # ===== Prueba Rápida =====
    logger.info("--- Prueba Rápida ---")

    test_text = "alice was talking to the cat in the garden"
    logger.info(f"Frase: {test_text}")

    entities = model.predict_entities(test_text, tokenizer)
    for txt, ent_type in entities:
        t = "Persona" if ent_type == "p" else "Lugar"
        logger.info(f"  [{txt}] -> {t}")
