import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module): # un módulo de pyTorch es un modelo entrenable

    def __init__(self, d_model, n_tokens, n_heads): # hiperparámetros
        # parámetros
        # self.W_qs = [nn.Linear(d_model, d_model) for _ in range(n_heads)] # dimensión de entrada y de salida
        # self.W_k = nn.Linear(d_model, d_model)
        # self.W_v = nn.Linear(d_model, d_model)

        self.qkv = nn.Linear(d_model, 3*d_model) # mejor con una supermatriz

        self.dropout = nn.Dropout(0.1) # % de neuronas que se desactivan

    def forward(self, x): # capa nueva de pyTorch, forward hace el cálculo hacia delante
        # x: tensor de los embeddings
        Q = self.W_q @ x # @: producto matricial
        K = self.W_k @ x # no hace falta trasponer la x
        V = self.W_v @ x
        A = Q @ K.transpose() # como K es un tensor, hay que decir en qué dimensiones trasponer
        A /= math.sqrt(self.d_model) # normalización
        A = F.softmax(A)
        A = self.dropout(A)
        return A @ V
    # retocar lo que falta en casa

    # falta la máscara

    #def backward(self) # cálculo hacia atrás