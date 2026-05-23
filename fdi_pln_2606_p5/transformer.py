"""Transformer básico con atención y feedforward."""

import torch
import torch.nn as nn

from attention import Attention


class FeedForward(nn.Module):
    """Capa feedforward del Transformer.

    Toma cada vector de embedding de un token y le aplica un perceptrón
    con capa intermedia *más amplia*, y activación GELU.

    El factor de expansión permite a la red encontrar y procesar patrones en un
    espacio menos denso que d_model.
    """

    def __init__(self, d_model, expansion, dropout):
        super().__init__()

        # Toma el vector de embedding de un token (d_model), lo expande a un
        # espacio 4 veces mayor (4 * d_model) y luego lo vuelve a comprimir a la dimensión original (d_model)
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),  # Función de activación no lineal
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Bloque de Transformer con atención y feedforward.

    Representa una capa completa ("Transformer Block") del modelo.
    Cada bloque tiene una capa de atención y una capa feedforward, con conexiones residuales y normalización.

    Incluye las dos cosas principales:
    1. Mecanismo de atención para atender al contexto, aprender matices y ambiguedades.
    2. Red feed-forward, para aprender a abstraer y generar las entradas de la siguiente capa.

    Se incluyen capas de normalización para regularizar el aprendizaje.
    """

    def __init__(self, d_model, n_heads, max_seq_len, expansion, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # Normalización
        self.attn = Attention(
            d_model, n_heads, max_seq_len, dropout
        )  # Capa de atención
        self.ln2 = nn.LayerNorm(d_model)  # Normalización
        self.ff = FeedForward(d_model, expansion, dropout)  # Capa feedforward

    def forward(self, x, causal=True):
        x = x + self.attn(
            self.ln1(x), causal=causal
        )  # Atención con conexión residual (llama a forward() de Attention)
        x = x + self.ff(
            self.ln2(x)
        )  # Feedforward con conexión residual (llama a forward() de FeedForward)
        return x


class Transformer(nn.Module):
    """Backbone del transformer: embeddings, bloques de atención y normalización final.

    Produce representaciones contextuales (hidden states) para cada token de
    entrada. No realiza ninguna tarea concreta: sirve de base para distintos
    modelos añadiéndole una cabeza específica (generación, clasificación…).

    Parámetros:
      vocab_size   Tamaño del vocabulario
      max_seq_len  Longitud máxima de secuencia
      d_model      Dimensión interna de las representaciones
      n_heads      Número de cabezas de atención
      n_layers     Número de bloques transformer apilados
      expansion    Factor de expansión de la capa feed-forward (típico: 4)
      dropout      Tasa de dropout para regularización
    """

    def __init__(
        self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # El inicio del transformer son los embeddings. Tenemos dos, los de
        # vocabulario y los posicionales (se calculan en forward)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Dropout para regularizar el aprendizaje de los embeddings
        self.drop = nn.Dropout(dropout)

        # El corazón del transformer es el bloque principal, con atención y
        # feedforward, que repetimos en secuencia varias veces
        self.blocks = nn.ModuleList(
            [
                Block(d_model, n_heads, max_seq_len, expansion, dropout)
                for _ in range(n_layers)
            ]
        )

        # Normalización final antes de la cabeza de salida
        self.norm = nn.LayerNorm(d_model)

    def forward(self, idx, causal=True):
        """Devuelve los hidden states para cada token de idx.

        idx     Tensor (batch, n_tokens) con ids de tokens
        causal  Si True, la atención es causal (solo mira tokens anteriores)
        """
        _, n_tokens = idx.shape

        # Embeddings de vocabulario + posicionales (se suman)
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(n_tokens, device=idx.device))
        x = self.drop(tok + pos)

        # Pasamos por todos los bloques transformer
        for block in self.blocks:
            x = block(x, causal=causal)

        # Normalización final
        return self.norm(x)
