"""Modelo NER, dataset y entrenamiento por fine-tuning del backbone Transformer.

Esquema de etiquetas (alineado con merged.json):
  o  → fuera de entidad (Outside)
  pi → inicio de persona  (Person-begin,  equiv. B-PER en BIO estándar)
  pc → continuación PER   (Person-cont,   equiv. I-PER)
  li → inicio de lugar    (Location-begin, equiv. B-LOC)
  lc → continuación LOC   (Location-cont,  equiv. I-LOC)

Flujo de datos:
  merged.json (tokens + etiquetas ya alineadas a BPE)
       |
       |  load_ner_from_merged()
       v
  lista de (ids_tensor, labels_tensor)
       |
       |  NERDataset + collate_ner
       v
  batches listos para cross_entropy
       |
       v
  NERLLM (Transformer backbone + cabeza lineal por token)
"""

import json
import time

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from transformer import Transformer

# Esquema de etiquetas del corpus Carroll
LABEL2ID = {"o": 0, "pi": 1, "pc": 2, "li": 3, "lc": 4}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Mapeo de etiqueta de inicio a tipo de entidad y su continuación
_ENTITY_START = {"pi": "PER", "li": "LOC"}
_ENTITY_CONT  = {"pi": "pc",  "li": "lc"}


# ---------------------------------------------------------------------------
# Alineamiento de etiquetas a sub-tokens BPE
# ---------------------------------------------------------------------------

def _align_word_to_bpe(word: str, label: str, tokenizer) -> tuple[list[int], list[str]]:
    """Codifica `word` con BPE y propaga la etiqueta a cada sub-token.

    Reglas:
      - 'pi' (inicio PER): primer sub-token 'pi', el resto 'pc'
      - 'li' (inicio LOC): primer sub-token 'li', el resto 'lc'
      - Cualquier otra etiqueta ('o', 'pc', 'lc'): se replica para todos los sub-tokens
    """
    ids = tokenizer.encode(word)
    if not ids:
        return [], []
    if label == "pi":
        labs = ["pi"] + ["pc"] * (len(ids) - 1)
    elif label == "li":
        labs = ["li"] + ["lc"] * (len(ids) - 1)
    else:
        labs = [label] * len(ids)
    return ids, labs


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_ner_from_merged(path, tokenizer, max_len: int = 128):
    """Carga merged.json y devuelve pares (ids_tensor, label_ids_tensor).

    Cada ítem tiene 'tokens' (lista de strings: palabras, espacios, puntuación)
    y sus 'labels' correspondientes.  El proceso para cada token:

      - Si el token está en el vocabulario BPE (tok2id): se usa su ID directamente
        y se conserva la etiqueta original.  Esto preserva la etiqueta del espacio
        entre tokens de entidad (ej. el ' ' entre 'march' y 'hare' tiene label 'pc').
      - Si no está en el vocabulario: se codifica con BPE carácter a carácter y se
        propaga la etiqueta con _align_word_to_bpe().

    IMPORTANTE: no filtramos espacios.  El espacio dentro de una entidad multi-
    palabra (ej. 'march hare') lleva etiqueta 'pc' en merged.json; si lo filtramos
    y reinsertamos como 'o' el modelo aprende a romper la entidad en el espacio.

    NOTA: Las frases de Carroll pueden tener hasta ~1100 tokens BPE; sin
    truncamiento el modelo falla al acceder a posiciones no existe en pos_emb.
    """
    data = json.load(open(path, encoding="utf-8"))
    samples = []

    for item in data:
        raw_tokens = item["tokens"]
        raw_labels = item["labels"]

        ids: list[int] = []
        labs: list[str] = []

        for tok_str, label in zip(raw_tokens, raw_labels):
            if len(ids) >= max_len:
                break

            if tok_str in tokenizer.tok2id:
                # Token exactamente en el vocabulario (espacio, carácter, merge frecuente)
                # → conservamos su etiqueta original (incluyendo espacios pc/lc)
                ids.append(tokenizer.tok2id[tok_str])
                labs.append(label)
            else:
                # Token (palabra o secuencia) no en el vocabulario
                # → re-tokenizamos con BPE y propagamos la etiqueta
                w_ids, w_labs = _align_word_to_bpe(tok_str, label, tokenizer)
                ids.extend(w_ids)
                labs.extend(w_labs)

        # Truncar al límite del modelo (crítico para los embeddings posicionales)
        ids  = ids[:max_len]
        labs = labs[:max_len]

        if not ids:
            continue

        label_ids = [LABEL2ID.get(l, 0) for l in labs]
        samples.append((
            torch.tensor(ids,       dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        ))

    return samples


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra → sub-tokens BPE para depuración.

    Útil para ver cómo la etiqueta B-/I- se distribuye al partir cada
    palabra en piezas BPE (la B- queda en el primer sub-token, el resto I-).
    """
    print(f"  frase: {' '.join(words)}")
    for word, label in zip(words, word_labels):
        ids, labs = _align_word_to_bpe(word, label, tokenizer)
        pieces = [tokenizer.decode([i]) for i in ids]
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


# ---------------------------------------------------------------------------
# Dataset y collate
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """Dataset de pares (ids_tensor, labels_tensor) listos para el modelo."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo máximo del batch.

    Las posiciones de padding llevan -100 en las etiquetas para que
    cross_entropy las ignore (ignore_index=-100).
    """
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
    return padded_x, padded_y


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER.

    Extiende Transformer añadiendo una capa lineal que asigna una etiqueta
    BIO a cada token.  Usa atención bidireccional (causal=False): para
    etiquetar un token podemos mirar el contexto a izquierda y derecha.

    Los pesos del backbone se inicializan desde un CausalLLM pre-entrenado
    usando load_state_dict(strict=False), que ignora las diferencias entre
    lm_head (generación) y ner_head (clasificación).
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        num_labels,
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None):
        """Devuelve (logits, loss).

        logits  (batch, n_tokens, num_labels)
        loss    cross-entropy sobre las posiciones no enmascaradas, o None
        """
        hidden = super().forward(input_ids, causal=False)
        logits = self.ner_head(hidden)

        if labels is not None:
            # Aplanamos batch y secuencia: cada token es una muestra.
            # ignore_index=-100 descarta posiciones de padding.
            loss = cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
            )
            return logits, loss

        return logits, None

    @torch.no_grad()
    def predict_entities(self, words, tokenizer):
        """Predice entidades nombradas a partir de una lista de palabras.

        IMPORTANTE: codifica cada palabra de forma aislada (igual que en el
        entrenamiento con merged.json), insertando el token de espacio entre
        palabras.  Codificar la oración completa produciría tokens distintos:
        ej. 'alice' + ' ' → 'alice ' (un único token) en lugar de
        ['alic', 'e'] + [' '] que el modelo aprendió a etiquetar.

        Devuelve [(texto_entidad, tipo), ...] donde tipo es 'PER' o 'LOC'.
        """
        self.eval()

        space_id = tokenizer.tok2id.get(" ", 0)
        ids: list[int] = []

        for i, word in enumerate(words):
            if i > 0:
                ids.append(space_id)
            word_ids = tokenizer.encode(word)
            ids.extend(word_ids)

        if not ids:
            return []

        # Respetar max_seq_len (embeddings posicionales limitados a esa longitud)
        ids = ids[: self.max_seq_len]

        device = next(self.parameters()).device
        logits, _ = self(torch.tensor([ids], device=device))
        pred_labels = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        # Palabras vacías que no pueden ser entidades nombradas
        _STOPWORDS = {"the", "a", "an", "of", "in", "at", "to", "and", "or",
                      "is", "was", "it", "he", "she", "they", "we", "you",
                      "said", "that", "this", "but", "so", "if", "on", "by"}

        entities = []
        i = 0
        while i < len(ids):
            lbl = pred_labels[i]
            if lbl in _ENTITY_START:
                kind = _ENTITY_START[lbl]
                cont = _ENTITY_CONT[lbl]
                j = i + 1
                while j < len(ids) and pred_labels[j] == cont:
                    j += 1
                span = tokenizer.decode(ids[i:j]).strip()
                # Filtrar fragmentos cortos y palabras vacías (probables FP)
                if span and len(span) >= 3 and span.lower() not in _STOPWORDS:
                    entities.append((span, kind))
                i = j
            else:
                i += 1

        return entities


# ---------------------------------------------------------------------------
# Entrenamiento NER
# ---------------------------------------------------------------------------

def _class_weights(ner_data, max_weight: float = 8.0) -> torch.Tensor:
    """Calcula pesos inversos a la frecuencia de cada etiqueta, con techo.

    El corpus Carroll tiene ~94% de tokens 'o' y solo ~6% de entidades.
    Sin pesos el modelo aprende a predecir 'o' para todo (alta accuracy,
    recall de entidades = 0).  Los pesos empujan la pérdida a ser mayor
    cuando falla en una entidad rara que cuando falla en un token común.

    Con max_weight limitamos pesos extremos (ej. 74x para 'li' con solo 16
    ejemplos) que harían al modelo sobre-detectar esa clase en todo token.
    """
    counts = torch.zeros(NUM_LABELS)
    for _, labs in ner_data:
        for l in labs.tolist():
            if 0 <= l < NUM_LABELS:
                counts[l] += 1
    total = counts.sum()
    # Peso inverso a la frecuencia, normalizado para que el mínimo sea 1.0
    weights = total / (NUM_LABELS * counts.clamp(min=1))
    weights = weights / weights.min()               # min_weight = 1.0
    weights = weights.clamp(max=max_weight)         # evitar extremos (ej. 74x → 8x)
    return weights


def train_ner(model, ner_data, epochs, lr, batch_size=16):
    """Fine-tune del modelo NER sobre los datos etiquetados de merged.json.

    Divide automáticamente en train/val (90/10), usa AdamW con weight decay,
    cosine annealing y pesos de clase para compensar el desequilibrio
    severo entre etiquetas 'o' y etiquetas de entidad (~96%/4%).
    """
    dataset = NERDataset(ner_data)

    # Split train/val — con corpus pequeño usamos mínimo 1 muestra de val
    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_ner)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_ner)

    logger.info(f"NER: {n_train} muestras train, {n_val} val")

    device  = next(model.parameters()).device
    weights = _class_weights(ner_data).to(device)
    logger.info(f"Pesos de clase: { {ID2LABEL[i]: f'{w:.2f}' for i, w in enumerate(weights.tolist())} }")

    # cross_entropy con pesos de clase e ignore_index para padding (-100)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    # Cosine annealing: LR decrece suavemente para no destruir el backbone pre-entrenado
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    t0 = time.time()
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)               # logits sin loss interna
            loss = criterion(logits.flatten(0, 1), y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1

        # --- Val ---
        model.eval()
        val_loss, m = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.flatten(0, 1), y.flatten())
                val_loss += loss.item()
                m += 1

        scheduler.step()

        logger.info(
            f"NER Época {epoch + 1:>2}/{epochs} | "
            f"train={train_loss / n:.4f} | val={val_loss / max(m, 1):.4f} | "
            f"tiempo={time.time() - t0:.1f}s"
        )
