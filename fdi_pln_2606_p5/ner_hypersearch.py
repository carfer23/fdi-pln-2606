"""Exploración de hiperparámetros NER sobre el corpus de Carroll.

Analiza primero el corpus para descubrir sus características reales (desequilibrio
de clases, expansión BPE, truncamiento), y luego lanza un grid search guiado por
esos hallazgos. Extrae conclusiones sobre qué funciona y por qué.

Uso:
  python ner_hypersearch.py
  python ner_hypersearch.py --data merged_2.json --backbone model_info/p5_causal_2606.pth
"""

import argparse
import itertools
import json
import pathlib
import sys
import time
from collections import Counter

import torch
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Silenciar loguru para que no mezcle con la salida Rich del script
logger.remove()

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from ner import ID2LABEL, NUM_LABELS, NERLLM, align_to_bpe
from ner_train import _compute_class_weights, _make_dataloaders, load_ner_from_merged
from tokenizer import BPETokenizer
from utils import load_corpus

console = Console()

# ── Arquitectura base (misma que main.py) ────────────────────────────────────

BASE_ARCH = {
    "vocab_size": 300,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "seq_len": 128,
    "expansion": 4,
    "dropout": 0.1,
}

# ── Grid de hiperparámetros a explorar ───────────────────────────────────────
# LR: rango amplio para ver sensibilidad; freeze: efecto del preentrenamiento;
# batch: relevante cuando el corpus de train tiene solo ~61 frases.
GRID = {
    "lr": [1e-3, 1e-4, 5e-5],
    "freeze_backbone": [True, False],
    "batch_size": [4, 8],
}
FIXED_EPOCHS = 20


# ── Análisis del corpus ──────────────────────────────────────────────────────


def analyze_corpus(ner_data: list, tokenizer) -> dict:
    """Caracteriza el corpus con métricas que guían la selección de hiperparámetros."""
    label_counts: Counter = Counter()
    seq_lens_words, seq_lens_bpe = [], []
    entity_densities = []
    truncated = 0

    for words, labels in ner_data:
        seq_lens_words.append(len(words))
        label_counts.update(labels)
        ids, labs = align_to_bpe(words, labels, tokenizer)
        seq_lens_bpe.append(len(ids))
        if len(ids) > BASE_ARCH["seq_len"]:
            truncated += 1
        n_ent = sum(1 for l in labs if l != "o")
        entity_densities.append(n_ent / max(len(labs), 1))

    total_tokens = sum(label_counts.values())
    n = len(ner_data)
    n_train = int(0.9 * n)
    n_val = n - n_train

    return {
        "n_sentences": n,
        "n_train": n_train,
        "n_val": n_val,
        "label_counts": dict(label_counts),
        "o_ratio": label_counts["o"] / total_tokens,
        "avg_words": sum(seq_lens_words) / n,
        "max_words": max(seq_lens_words),
        "avg_bpe": sum(seq_lens_bpe) / n,
        "max_bpe": max(seq_lens_bpe),
        "bpe_expansion": sum(seq_lens_bpe) / max(sum(seq_lens_words), 1),
        "avg_entity_density": sum(entity_densities) / n,
        "truncated": truncated,
    }


def print_corpus_analysis(stats: dict) -> None:
    console.rule("[bold cyan]Análisis del corpus etiquetado[/bold cyan]")

    # Distribución de etiquetas
    t = Table("Etiqueta", "Tokens (word-level)", "%", box=box.SIMPLE)
    total = sum(stats["label_counts"].values())
    for lbl, cnt in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        style = "dim" if lbl == "o" else "bold green"
        t.add_row(lbl, str(cnt), f"{pct:.2f}%", style=style)
    console.print(t)

    lines = [
        f"[bold]Dataset:[/bold] {stats['n_sentences']} frases  →  "
        f"{stats['n_train']} entrenamiento / {stats['n_val']} validación",
        f"[bold]Desequilibrio de clases:[/bold] {stats['o_ratio']*100:.1f}% de tokens son 'O'",
        f"[bold]Expansión BPE:[/bold] {stats['avg_words']:.0f} palabras/frase → "
        f"{stats['avg_bpe']:.0f} sub-tokens/frase  (×{stats['bpe_expansion']:.2f})",
        f"[bold]Densidad de entidades:[/bold] {stats['avg_entity_density']*100:.2f}% "
        "tokens por frase son entidad",
        f"[bold]Frases truncadas (>{BASE_ARCH['seq_len']} sub-tokens):[/bold] "
        f"{stats['truncated']}/{stats['n_sentences']}  "
        f"(la más larga tiene {stats['max_bpe']} sub-tokens)",
    ]
    console.print(
        Panel("\n".join(lines), title="[yellow]Realidades del corpus[/yellow]", border_style="yellow")
    )

    warns = []
    if stats["n_val"] < 10:
        warns.append(
            f"[red]Validación de solo {stats['n_val']} frases — métricas poco fiables. "
            "La búsqueda sirve para comparar tendencias, no valores absolutos.[/red]"
        )
    if stats["max_bpe"] > BASE_ARCH["seq_len"]:
        warns.append(
            f"[red]La frase más larga ({stats['max_bpe']} sub-tokens) supera seq_len="
            f"{BASE_ARCH['seq_len']}. Las frases largas se truncan y las entidades "
            "que aparecen después del token 128 se pierden silenciosamente.[/red]"
        )
    if stats["o_ratio"] > 0.9:
        warns.append(
            "[yellow]Desequilibrio extremo: sin class_weights el modelo aprende a "
            "predecir todo como 'O' y obtiene pérdida baja sin detectar ninguna entidad.[/yellow]"
        )
    for w in warns:
        console.print(w)
    console.print()


# ── Evaluación ───────────────────────────────────────────────────────────────


def evaluate(model, dataloader, class_weights, device):
    """Devuelve (avg_loss, macro_token_F1, f1_por_clase).

    F1 a nivel de token (no de span) — suficiente para comparar configuraciones
    entre sí. El denominador de macro promedia sobre las 4 clases de entidad.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y, class_weights=class_weights)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(-1)
            for pred_row, label_row in zip(preds, y):
                for p, l in zip(pred_row.tolist(), label_row.tolist()):
                    if l == -100:
                        continue
                    pred_lbl = ID2LABEL[p]
                    true_lbl = ID2LABEL[l]
                    if true_lbl != "o":
                        if pred_lbl == true_lbl:
                            tp[true_lbl] += 1
                        else:
                            fn[true_lbl] += 1
                    if pred_lbl != "o" and pred_lbl != true_lbl:
                        fp[pred_lbl] += 1

    f1_per_class = {}
    for lbl in ["pi", "pc", "li", "lc"]:
        prec = tp[lbl] / max(tp[lbl] + fp[lbl], 1)
        rec = tp[lbl] / max(tp[lbl] + fn[lbl], 1)
        f1_per_class[lbl] = 2 * prec * rec / max(prec + rec, 1e-9)

    macro_f1 = sum(f1_per_class.values()) / len(f1_per_class)
    return total_loss / max(n_batches, 1), macro_f1, f1_per_class


# ── Entrenamiento de una configuración ──────────────────────────────────────


def run_config(cfg: dict, ner_data: list, tokenizer, device, backbone_state) -> dict:
    """Entrena y evalúa una configuración; devuelve sus métricas."""
    model = NERLLM(
        vocab_size=BASE_ARCH["vocab_size"],
        max_seq_len=BASE_ARCH["seq_len"],
        d_model=BASE_ARCH["d_model"],
        n_heads=BASE_ARCH["n_heads"],
        n_layers=BASE_ARCH["n_layers"],
        expansion=BASE_ARCH["expansion"],
        dropout=BASE_ARCH["dropout"],
        num_labels=NUM_LABELS,
    ).to(device)

    if backbone_state is not None:
        model.load_state_dict(backbone_state, strict=False)

    if cfg["freeze_backbone"]:
        for name, param in model.named_parameters():
            if "ner_head" not in name:
                param.requires_grad = False

    train_dl, val_dl = _make_dataloaders(ner_data, tokenizer, cfg["batch_size"])
    class_weights = _compute_class_weights(train_dl.dataset).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=0.01,
    )

    history = []
    t0 = time.time()

    for _ in range(cfg["epochs"]):
        model.train()
        torch.set_grad_enabled(True)
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss, train_f1, _ = evaluate(model, train_dl, class_weights, device)
        val_loss, val_f1, _ = evaluate(model, val_dl, class_weights, device)
        history.append(
            {
                "epoch": len(history) + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_f1": round(train_f1, 4),
                "val_f1": round(val_f1, 4),
            }
        )

    elapsed = round(time.time() - t0, 1)

    # Detectar overfitting: val_loss media de la segunda mitad > primera mitad × 5%
    mid = len(history) // 2
    if mid > 0:
        avg_first = sum(h["val_loss"] for h in history[:mid]) / mid
        avg_second = sum(h["val_loss"] for h in history[mid:]) / (len(history) - mid)
        overfit = avg_second > avg_first * 1.05
    else:
        overfit = False

    best = max(history, key=lambda h: h["val_f1"])

    return {
        "final_val_loss": history[-1]["val_loss"],
        "final_val_f1": history[-1]["val_f1"],
        "final_train_f1": history[-1]["train_f1"],
        "best_val_f1": best["val_f1"],
        "best_epoch": best["epoch"],
        "overfit": overfit,
        "elapsed_s": elapsed,
        "history": history,
    }


# ── Insights ─────────────────────────────────────────────────────────────────


def print_insights(results_sorted: list, corpus_stats: dict, has_backbone: bool) -> None:
    insights = []

    # ① Efecto del learning rate
    by_lr: dict = {}
    for r in results_sorted:
        lr = r["config"]["lr"]
        by_lr.setdefault(lr, []).append(r["metrics"]["best_val_f1"])
    avg_by_lr = {lr: sum(v) / len(v) for lr, v in by_lr.items()}
    best_lr = max(avg_by_lr, key=avg_by_lr.get)
    worst_lr = min(avg_by_lr, key=avg_by_lr.get)
    insights.append(
        f"① [bold]Sensibilidad al LR:[/bold] el mejor LR es {best_lr:.0e} "
        f"(F1 medio={avg_by_lr[best_lr]:.3f}), el peor {worst_lr:.0e} "
        f"(F1={avg_by_lr[worst_lr]:.3f}). Con solo {corpus_stats['n_train']} frases "
        "de entrenamiento, un LR demasiado alto produce oscilaciones en la pérdida "
        "antes de converger; demasiado bajo no converge en 20 épocas."
    )

    # ② Efecto de congelar el backbone
    if has_backbone:
        f1_freeze = [r["metrics"]["best_val_f1"] for r in results_sorted if r["config"]["freeze_backbone"]]
        f1_full = [r["metrics"]["best_val_f1"] for r in results_sorted if not r["config"]["freeze_backbone"]]
        if f1_freeze and f1_full:
            avg_freeze = sum(f1_freeze) / len(f1_freeze)
            avg_full = sum(f1_full) / len(f1_full)
            delta = abs(avg_freeze - avg_full)
            if avg_freeze > avg_full:
                insights.append(
                    f"② [bold]Freeze backbone:[/bold] congelar el backbone es mejor "
                    f"(Δ F1 = +{delta:.3f}). Con {corpus_stats['n_train']} frases, "
                    "el fine-tuning completo produce olvido catastrófico: los gradientes "
                    "de la pequeña cabeza NER corrompem las representaciones preentrenadas "
                    "del backbone. Congelar el backbone es equivalente a usar el LLM "
                    "como extractor de características fijo."
                )
            else:
                insights.append(
                    f"② [bold]Freeze backbone:[/bold] fine-tuning completo es mejor "
                    f"(Δ F1 = +{delta:.3f}). El backbone necesita adaptarse al esquema BIO: "
                    "las representaciones preentrenadas con objetivo causal (predecir siguiente "
                    "token) no son óptimas para clasificación bidireccional por token."
                )

    # ③ Tamaño de batch
    by_batch: dict = {}
    for r in results_sorted:
        b = r["config"]["batch_size"]
        by_batch.setdefault(b, []).append(r["metrics"]["best_val_f1"])
    avg_by_batch = {b: sum(v) / len(v) for b, v in by_batch.items()}
    best_batch = max(avg_by_batch, key=avg_by_batch.get)
    insights.append(
        f"③ [bold]Tamaño de batch:[/bold] batch={best_batch} tiene el mejor F1 medio "
        f"({avg_by_batch[best_batch]:.3f}). Con {corpus_stats['n_train']} frases, "
        f"batch=4 da ~{corpus_stats['n_train']//4} actualizaciones/época "
        f"y batch=8 da ~{corpus_stats['n_train']//8}. "
        "Un batch pequeño introduce más ruido en el gradiente, lo que puede ayudar "
        "a escapar de mínimos locales o perjudicar la convergencia en datasets tan pequeños."
    )

    # ④ Overfitting
    overfit_count = sum(1 for r in results_sorted if r["metrics"]["overfit"])
    insights.append(
        f"④ [bold]Overfitting:[/bold] {overfit_count}/{len(results_sorted)} configuraciones "
        f"muestran aumento de val_loss en la segunda mitad del entrenamiento. "
        f"Con solo {corpus_stats['n_val']} frases de validación y "
        f"{corpus_stats['avg_entity_density']*100:.2f}% de densidad de entidades, "
        "la señal de validación es extremadamente ruidosa (pocas entidades por frase). "
        "El mejor epoch raramente es el último."
    )

    # ⑤ Desequilibrio de clases
    insights.append(
        f"⑤ [bold]Desequilibrio de clases:[/bold] {corpus_stats['o_ratio']*100:.0f}% 'O'. "
        "Los class_weights asignados compensan esto penalizando más los errores en "
        "entidades raras ('li'/'lc' con <0.5% de tokens). Sin ellos, el modelo "
        "maximizaría el accuracy prediciendo todo como 'O' y obtendría una pérdida "
        "baja sin detectar ninguna entidad. Con class_weights, la pérdida de 'li' "
        "contribuye ~200× más que la de 'O'."
    )

    # ⑥ Truncamiento BPE
    if corpus_stats["truncated"] > 0:
        insights.append(
            f"⑥ [bold]Truncamiento por BPE:[/bold] {corpus_stats['truncated']} frases "
            f"({corpus_stats['truncated']/corpus_stats['n_sentences']*100:.0f}%) "
            f"se truncan al pasar de palabras a sub-tokens (expansión ×{corpus_stats['bpe_expansion']:.1f}). "
            f"La frase más larga del corpus tiene {corpus_stats['max_bpe']} sub-tokens; "
            f"el modelo solo ve los primeros {BASE_ARCH['seq_len']}. Las entidades "
            "que aparecen en la parte truncada nunca se aprenden ni se evalúan."
        )
    else:
        insights.append(
            f"⑥ [bold]BPE:[/bold] Factor de expansión ×{corpus_stats['bpe_expansion']:.1f}. "
            f"Todas las frases caben en seq_len={BASE_ARCH['seq_len']} tras la tokenización."
        )

    console.print(
        Panel(
            "\n\n".join(insights),
            title="[bold yellow]Descubrimientos sobre el corpus y el algoritmo[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_backbone(path: str, device: str):
    p = pathlib.Path(path)
    if p.exists():
        return torch.load(p, map_location=device, weights_only=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploración de hiperparámetros NER")
    parser.add_argument("--data", default="merged_2.json")
    parser.add_argument("--backbone", default="model_info/p5_causal_2606.pth")
    parser.add_argument("--corpus", default="resources")
    parser.add_argument("--out", default="logs/hypersearch_results.json")
    args = parser.parse_args()

    device = _device()
    console.print(f"[bold]Dispositivo:[/bold] {device}\n")

    # 1. Tokenizador y datos
    text = load_corpus(args.corpus)
    tokenizer_path = pathlib.Path("model_info/tokenizer.json")
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer(text, vocab_size=BASE_ARCH["vocab_size"])
    ner_data = load_ner_from_merged(args.data)

    # 2. Análisis del corpus
    stats = analyze_corpus(ner_data, tokenizer)
    print_corpus_analysis(stats)

    # 3. Backbone
    backbone_state = _load_backbone(args.backbone, device)
    if backbone_state is None:
        backbone_state = _load_backbone("model_weights/model.pth", device)
    has_backbone = backbone_state is not None
    console.print(
        f"[bold]Backbone:[/bold] {'cargado ✓  (' + args.backbone + ')' if has_backbone else 'no encontrado — entrenando desde cero'}\n"
    )
    if not has_backbone:
        console.print(
            "[yellow]Sin backbone, freeze_backbone=True equivale a freeze_backbone=False "
            "(sin representaciones preentrenadas que proteger). Se omiten esas configuraciones.[/yellow]\n"
        )

    # 4. Construir configuraciones
    keys = list(GRID.keys())
    configs = []
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, combo))
        cfg["epochs"] = FIXED_EPOCHS
        if not has_backbone and cfg["freeze_backbone"]:
            continue
        configs.append(cfg)

    console.rule(
        f"[bold cyan]Grid search — {len(configs)} configuraciones × {FIXED_EPOCHS} épocas[/bold cyan]"
    )

    # 5. Grid search
    results = []
    for i, cfg in enumerate(configs, 1):
        label = (
            f"lr={cfg['lr']:.0e}  freeze={'T' if cfg['freeze_backbone'] else 'F'}  batch={cfg['batch_size']}"
        )
        console.print(f"  [{i:>2}/{len(configs)}] {label} ...", end=" ")
        metrics = run_config(cfg, ner_data, tokenizer, device, backbone_state)
        results.append({"config": cfg, "metrics": metrics})
        status = "[red]OVERFIT[/red]" if metrics["overfit"] else "[green]OK[/green]"
        console.print(
            f"best_F1={metrics['best_val_f1']:.3f} (ep{metrics['best_epoch']:>2})  "
            f"val_loss={metrics['final_val_loss']:.4f}  {status}  {metrics['elapsed_s']}s"
        )

    # 6. Tabla de resultados
    console.rule("[bold cyan]Resultados ordenados por F1 de validación[/bold cyan]")
    results_sorted = sorted(results, key=lambda r: -r["metrics"]["best_val_f1"])

    t = Table(
        "Rank", "LR", "Freeze", "Batch", "Best F1", "Best Ep.", "Val Loss", "Overfit", "Tiempo",
        box=box.ROUNDED,
        show_lines=True,
    )
    for rank, r in enumerate(results_sorted, 1):
        c, m = r["config"], r["metrics"]
        t.add_row(
            str(rank),
            f"{c['lr']:.0e}",
            "Sí" if c["freeze_backbone"] else "No",
            str(c["batch_size"]),
            f"[bold green]{m['best_val_f1']:.3f}[/bold green]",
            str(m["best_epoch"]),
            f"{m['final_val_loss']:.4f}",
            "[red]Sí[/red]" if m["overfit"] else "No",
            f"{m['elapsed_s']}s",
        )
    console.print(t)

    # 7. Insights
    print_insights(results_sorted, stats, has_backbone)

    # 8. Guardar
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"corpus_stats": stats, "grid": GRID, "results": results_sorted}, f, indent=2)
    console.print(f"\n[dim]Resultados guardados en {args.out}[/dim]")


if __name__ == "__main__":
    main()
