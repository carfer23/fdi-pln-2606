import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax

from attention import Attention

class FeedForward(nn.Module):
    """Capa feedforward del Transformer.

    Toma cada vector de embedding de un token y le aplica un perceptrón.
    """

    def __init__(self, d_model, expansion, dropout):
        super().__init__()
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Bloque de Transformer con atención y feedforward.
    
    Representa una capa completa ("Transformer Block") del modelo.
    Cada bloque tiene una capa de atención y una capa feedforward, con conexiones residuales y normalización.
    """

    def __init__(self, d_model, n_heads, max_seq_len, expansion, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, expansion, dropout)

    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal=causal) # Atención con conexión residual (llama a forward() de Attention)
        x = x + self.ff(self.ln2(x)) # Feedforward con conexión residual (llama a forward() de FeedForward)
        return x
        

class Transformer(nn.Module):
    """Backbone del transformer: embeddings, bloques de atención y normalización final.

    Produce hidden states para cada token. No realiza ninguna tarea concreta:
    sirve de base para añadirle una cabeza específica (generación, clasificación…).
    """

    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, max_seq_len, expansion, dropout) for _ in range(n_layers)]
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

        return self.norm(x)


class CausalLLM(Transformer):
    """Modelo de lenguaje causal para generación de texto.

    Añade una cabeza lineal que proyecta los hidden states al vocabulario.
    Usa weight tying: los pesos del embedding de entrada y la cabeza de salida
    son los mismos, lo que mejora la generalización y reduce parámetros.
    """

    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout):
        super().__init__(vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        """Devuelve (logits, loss).

        idx      Tensor (batch, n_tokens) con ids de tokens de entrada
        targets  Tensor (batch, n_tokens) con ids objetivo; si se pasa, calcula el loss
        """
        x = super().forward(idx, causal=True)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        # Aplanamos el batch para calcular cross-entropy eficientemente
        loss = cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8):
        """Genera tokens a partir de un prompt (lista de ids).

        prompt       Lista de token ids de entrada.
        max_tokens   Número máximo de tokens a generar.
        temperature  Modula lo "puntiaguda" (determinista) que es la distribución.
        """
        self.eval()

        # Ventana deslizante inicializada con el prompt, con dimensión batch=1
        ventana = torch.tensor(
            [prompt[-self.max_seq_len:]],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        generados = []
        for _ in range(max_tokens):
            logits, _ = self(ventana)
            next_token_logits = logits[:, -1, :]
            # Temperatura: divide los logits antes del softmax para modular la aleatoriedad
            next_token_probs = softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, 1)

            generados.append(next_token_id.item())
            # Deslizamos la ventana descartando el token más antiguo si es necesario
            ventana = torch.cat([ventana, next_token_id], dim=1)[:, -self.max_seq_len:]

        return generados