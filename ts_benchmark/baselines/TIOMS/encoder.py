import torch
import torch.nn as nn
import torch.nn.functional as F


class OHLCVFeatureEncoder(nn.Module):
    """
    Processa features locais OHLCV de cada papel, preservando:
        - B: batch
        - T: tempo
        - N: canais = papéis

    Entrada:
        x: (B, T, N, F)   com F = 5 por padrão

    Saída:
        z: (B, T, N, d_model)

    Ideia:
        Para cada (b, t, n), aplica o MESMO encoder em x[b, t, n, :].
        Ou seja, funde OHLCV intra-papel, sem misturar papéis entre si.
    """

    def __init__(
        self,
        in_features: int = 7,
        d_model: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual and (in_features == d_model)

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # Gate opcional para dosar importância das features OHLCV
        self.gate = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aceita:
            x: (B, T, N, F)  ou  (M, T, F)

        Retorna:
            (B, T, N, d_model)  ou  (M, T, d_model)
        """
        if x.dim() not in {3, 4}:
            raise ValueError(
                f"Esperado x com 3 ou 4 dimensões ((M,T,F) ou (B,T,N,F)), mas veio {tuple(x.shape)}"
            )

        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Última dimensão deveria ser {self.in_features}, mas veio {x.size(-1)}"
            )

        g = torch.sigmoid(self.gate(x))
        x = x * g

        z = self.fc1(x)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.fc2(z)

        if self.use_residual:
            z = z + x

        z = self.norm(z)
        return z