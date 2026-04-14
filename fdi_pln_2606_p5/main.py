import torch.nn as nn

from atteention import Attention

class FeedForward(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Lienar(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ff(self.ln2(x))
        return x
        
#class MiniLLM(nn.Module):