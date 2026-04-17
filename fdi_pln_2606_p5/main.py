import torch.nn as nn

from attention import Attention

class FeedForward(nn.Module):
    """Capa feedforward del Transformer.

    Toma cada vector de embedding de un token y le aplica un perceptrón.
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        # Toma el vector de embedding de un token (d_model), lo expande a un espacio 4 veces mayor (4 * d_model) 
        # y luego lo vuelve a comprimir a la dimensión original
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(), # Función de activación no lineal
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Bloque de Transformer con atención y feedforward.
    
    Representa una capa completa ("Transformer Block") del modelo.
    Cada bloque tiene una capa de atención y una capa feedforward, con conexiones residuales y normalización.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) # Normalización
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout) # Mecanismo de atención
        self.ln2 = nn.LayerNorm(d_model) # Normalización
        self.ff = FeedForward(d_model, dropout) # Capa feedforward

    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal=causal) # Atención con conexión residual (llama a forward() de Attention)
        x = x + self.ff(self.ln2(x)) # Feedforward con conexión residual (llama a forward() de FeedForward)
        return x
        
#class MiniLLM(nn.Module):