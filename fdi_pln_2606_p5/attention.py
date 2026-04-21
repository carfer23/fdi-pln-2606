"""Mecanismo de atención"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Attention(nn.Module): # Un módulo de PyTorch es un modelo entrenable
    """Auto-atención multi-cabezal con escala (scaled multi-head self-attention)

    Si `causal=True` en el forward, cada posición solo atiende a las
    anteriores (útil para generación). Si `causal=False`, cada posición
    atiende a toda la secuencia (TAREA: para qué querríamos esto?).

    dropout es el porcentaje de dropout a usar.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout): # Hiperparámetros
        # Parámetros
        super().__init__()

        self.n_heads = n_heads # Número de cabezas de atención
        self.head_dim = d_model // n_heads  # dimensión por cabezal

        # Matrices separadas
        # self.W_qs = [nn.Linear(d_model, d_model) for _ in range(n_heads)] # dimensión de entrada y de salida
        # self.W_k = nn.Linear(d_model, d_model)
        # self.W_v = nn.Linear(d_model, d_model)

        # Una única matriz para QKV, luego separaremos
        # nn.Lineal -> capa lineal, toma un vector de tamaño d_model y lo proyecta a un espacio 3 veces más grande
        # PyTorch se encarga de inicializar la matriz de pesos de esta capa lineal de forma aleatoria
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Capa lineal para permitir al modelo reproyectar los vectores contexto
        # TAREA: necesaria? y si la quito?
        self.out = nn.Linear(d_model, d_model)

        # El dropout se activa en train y desactiva en test gracias a pytorch
        self.dropout = nn.Dropout(dropout) # % de neuronas que se desactivan

        # La máscara causal pone a -inf las posiciones correspondientes a tokens
        # "futuros" (triangular superior)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )
        # Registramos la máscara causal como tensor (no entrenable)
        self.register_buffer("mask", mask)

    def forward(self, x, causal=True): # forward hace el cálculo hacia delante
        """
        :param x: tensor de los embeddings. Shape (batch_size, n_tokens, d_model)
        :param causal: si True, se aplica la máscara causal para que cada posición solo atienda a las anteriores.
        :return: tensor con la misma shape que x, pero con la información de contexto añadida.
        """
        # Los tensores de pytorch tienen primero una dimensión batch
        # (entrenamiento más eficiente si hacemos varios a la vez)
        # luego tokens y luego ya la dimensión de los embeddings
        # batch_size, n_tokens, d_model = x.shape

        # Q = self.W_q @ x # @: producto matricial
        # K = self.W_k @ x # no hace falta trasponer la x
        # V = self.W_v @ x

        # multiplicamos x por QKV (todo junto)
        # separamos por la última dimensión para tener las 3 matrices de queries, keys y values
        q, k, v = self.qkv(x).tensor_split(3, dim=-1)

        # separamos en cabezales (ver función más abajo)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        a = q @ k.transpose(-2, -1) # (batch_size, n_heads, n_tokens, head_dim) @ (batch_size, n_heads, head_dim, n_tokens) -> (batch_size, n_heads, n_tokens, n_tokens)
        # Nota: para escalar, dividir por raíz de head_dim (para que los logits no crezcan sin control)
        a /= math.sqrt(self.head_dim)

        if causal:
            seq_len = x.shape[1] # n_tokens
            # Al sumar la máscara, las posiciones futuras adquieren valor -inf
            a = a + self.mask[:seq_len, :seq_len]

        a = F.softmax(a, dim=-1)
        a = self.dropout(a)
        z = a @ v # z queda con shape (batch_size, n_heads, n_tokens, head_dim)

        # "deshacemos" la partición en cabezales
        # (batch_size, n_heads, n_tokens, head_dim) -> (batch_size, n_tokens, d_model)
        z = z.transpose(1, 2).flatten(-2)

        # re-proyectamos con la última transformación
        return self.out(z)

    def split_heads(self, x):
        """Separa el tensor x (Q, K o V) en n_heads cabezales de atención."""
        # 1. (batch_size, n_tokens, d_model) -> (batch_size, n_tokens, n_heads, head_dim)
        # "partimos" la última dimensión (-1, d_model) en n_heads vectores de tamaño head_dim
        x = x.unflatten(-1, (self.n_heads, self.head_dim))

        # 2. (batch_size, n_tokens, n_heads, head_dim) -> (batch_size, n_heads, n_tokens, head_dim)
        # transponemos n_tokens y n_heads para que cada cabezal de atención se
        # "multiplique por separado", haciéndolos independientes
        # TAREA: hacer el álgebra a mano para ver como funciona
        return x.transpose(1, 2)