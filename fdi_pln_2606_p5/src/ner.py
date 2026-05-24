"""Modelo NER, dataset y alineamiento palabra -> BPE.

Esquema de etiquetas:
  o  → fuera de entidad
  pi → inicio de persona
  pc → continuación persona
  li → inicio de lugar
  lc → continuación lugar

El flujo de datos:

  palabras anotadas (word-level, BIO)
        |
        |  align_to_bpe()       <- asigna B-/I- a cada sub-token
        v
  sub-tokens + etiquetas BIO
        |
        |  NERDataset + collate_ner
        v
  batches (ids, labels) listos para cross_entropy
        |
        v
  NERLLM (Transformer + cabeza lineal por token)
"""

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

from transformer import Transformer

# Etiquetas NER en esquema BIO
LABEL2ID = {"o": 0, "pi": 1, "pc": 2, "li": 3, "lc": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def align_to_bpe(words, word_labels, tokenizer):
    """Alinea etiquetas NER palabra -> sub-tokens BPE.

    `words` puede incluir espacios y puntuacion (tal cual vienen del JSON),
    por lo que NO se insertan espacios extra. Cada token del JSON se tokeniza
    de forma independiente y sus sub-tokens heredan la etiqueta.

    Regla: la etiqueta inicial (terminada en 'i') se queda en el primer
    sub-token y los siguientes son continuacion (terminada en 'c').

      palabra 'alice' con etiqueta pi, BPE la parte en ['al', 'ice']
         -> al: pi, ice: pc
      palabra 'cheshire' con etiqueta pc, BPE en ['ch', 'es', 'h', 'ire']
         -> todos pc
      palabra 'wonderland' con etiqueta li, BPE en ['won', 'der', 'land']
         -> won: li, der: lc, land: lc
      tokens con etiqueta o -> todos sus sub-tokens o

    Devuelve (token_ids, token_labels) con etiquetas como strings.
    """
    token_ids = []
    token_labels = []
    for word, label in zip(words, word_labels):
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        if label.endswith("i"):
            inside = label[0] + "c"
            token_labels.append(label)
            token_labels.extend([inside] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))
    return token_ids, token_labels


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra -> sub-tokens BPE para una frase.

    Util para ver como el tokenizador parte cada token (incluidos espacios
    y puntuacion) y donde aterriza cada etiqueta.
    """
    print("  frase:")
    print("".join(words))
    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        if label.endswith("i"):
            inside = label[0] + "c"
            labs = [label] + [inside] * (len(ids) - 1)
        else:
            labs = [label] * len(ids)
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER.

    Extiende Transformer añadiendo una cabeza lineal que asigna una etiqueta
    a cada token. Usa atención bidireccional (causal=False): para etiquetar
    un token podemos mirar el contexto a derecha e izquierda.

    Los pesos del backbone se deben inicializar desde un CausalLLM pre-entrenado
    con load_state_dict(strict=False), que ignora las diferencias en las cabezas
    (lm_head vs ner_head) y transfiere solo el backbone compartido.
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
        # El transformer ya tiene una representación suficientemente rica,
        # no tenemos más que proyectarla al espacio de etiquetas
        self.ner_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels=None, class_weights=None):
        hidden = super().forward(input_ids, causal=False)
        logits = self.ner_head(hidden)
        loss = None
        if labels is not None:
            # cross_entropy espera logits 2D: para cada elemento, una
            # probabilidad por etiqueta.
            # Aplanamos batch y secuencia y tratamos cada token como una muestra
            # independiente:
            #   logits  (n_batches, n_tokens, num_labels) -> (n_batches*n_tokens, num_labels)
            #   labels  (n_batches, n_tokens)             -> (n_batches*n_tokens,)
            # Las posiciones de padding llevan -100 e ignore_index las descarta.
            flat_logits = logits.flatten(0, 1)
            flat_labels = labels.flatten()
            loss = cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                weight=class_weights,
            )
        return logits, loss

    @torch.no_grad()
    def predict_entities(self, text, tokenizer):
        """Predice entidades NER a partir de texto crudo.

        El texto se codifica con `tokenizer.encode`, se corre el modelo y se
        agrupan sub-tokens iniciales y de continuacion consecutivos en entidades.

        Devuelve las entidades nombradas ya compuestas [(texto, tipo), ...].
        """
        self.eval()
        ids = tokenizer.encode(text)
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor([ids], device=device))
        pred_labels = [ID2LABEL[p] for p in logits.argmax(-1)[0].tolist()]

        entities = []
        i = 0
        while i < len(ids):
            if pred_labels[i] != "o":
                kind = pred_labels[i][0]  # Extrae 'p' de 'pi'/'pc' o 'l' de 'li'/'lc'
                j = i + 1
                while j < len(ids) and pred_labels[j] == f"{kind}c":
                    j += 1
                ent_text = tokenizer.decode(ids[i:j]).strip()
                if ent_text:
                    entities.append((ent_text, kind))
                i = j
            else:
                i += 1
        return entities


class NERDataset(Dataset):
    """Dataset de NER: aplica `align_to_bpe` a cada frase y convierte a tensores.

    `ner_data` es una lista de pares (words, labels), donde words es la lista
    de palabras de una frase y labels las etiquetas alineadas. Es el
    formato que produce `load_ner_data` al leer el TSV en formato CoNLL.
    """

    def __init__(self, ner_data, tokenizer, max_len=128):
        self.samples = []
        for words, labels in ner_data:
            ids, labs = align_to_bpe(words, labels, tokenizer)
            ids = ids[:max_len]
            labs = labs[:max_len]
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor([LABEL2ID[l] for l in labs], dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo maximo del batch. Las posiciones de padding usan -100
    en las etiquetas para que cross_entropy las ignore (no son tokens reales)."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
    return padded_x, padded_y
