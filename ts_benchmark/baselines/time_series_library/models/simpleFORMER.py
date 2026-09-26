"""Direct forecaster with joint time tokens and temporal self-attention.

There is one token per historical instant, not one token per asset. Attention
projections are shared across instants; layers have independent parameters.
"""
import math

import torch
from torch import nn

from .simple_common import SimpleForecastBase, positive_int


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normalized = self.attention_norm(x)
        # All tokens belong to the observed lookback: no causal mask required.
        # Explicit attention weights avoid fused attention backend differences.
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=True)
        x = x + self.attention_dropout(attended)
        return x + self.ffn(self.ffn_norm(x))


class simpleFORMER(SimpleForecastBase):
    def __init__(self, configs):
        super().__init__(configs)
        n_heads = positive_int(configs, "n_heads", 8)
        if self.d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        d_ff = positive_int(configs, "d_ff", 4 * self.d_model)
        positions = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros(1, self.seq_len, self.d_model)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(
            positions * frequencies[: self.d_model // 2]
        )
        self.register_buffer("position_encoding", encoding)
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        self.blocks = nn.ModuleList([
            TemporalAttentionBlock(self.d_model, n_heads, d_ff, self.dropout_rate)
            for _ in range(self.e_layers)
        ])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        features, mean, scale = self.encode_input(x_enc)
        features = self.embedding_dropout(
            features + self.position_encoding.to(dtype=features.dtype)
        )
        for block in self.blocks:
            features = block(features)
        return self.decode_output(features, mean, scale)
