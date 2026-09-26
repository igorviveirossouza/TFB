"""Joint channel embedding followed by shared residual temporal MLPs."""
from torch import nn

from .simple_common import SimpleForecastBase


class TemporalMLPBlock(nn.Module):
    def __init__(self, seq_len, d_model, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # The same temporal weights process each latent feature; not N networks.
        return x + self.mlp(self.norm(x).transpose(1, 2)).transpose(1, 2)


class simpleMLP(SimpleForecastBase):
    def __init__(self, configs):
        super().__init__(configs)
        hidden_dim = getattr(configs, "temporal_hidden_dim", 0)
        if isinstance(hidden_dim, bool) or int(hidden_dim) != hidden_dim or hidden_dim < 0:
            raise ValueError("temporal_hidden_dim must be an integer >= 0")
        # Zero means automatic: LB -> 2*LB -> LB.
        hidden_dim = int(hidden_dim) or 2 * self.seq_len
        self.blocks = nn.ModuleList([
            TemporalMLPBlock(
                self.seq_len, self.d_model, hidden_dim, self.dropout_rate
            )
            for _ in range(self.e_layers)
        ])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        features, mean, scale = self.encode_input(x_enc)
        for block in self.blocks:
            features = block(features)
        return self.decode_output(features, mean, scale)
