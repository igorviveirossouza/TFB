import torch
import torch.nn as nn


class SpectralBandDecomposer(nn.Module):
    """
    Decomposição espectral em três bandas:
    baixa, média e alta frequência.

    Entrada:
        x: (B, T, N)

    Saída:
        x_low:  (B, T, N)
        x_mid:  (B, T, N)
        x_high: (B, T, N)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape

        # rFFT ao longo da dimensão temporal
        # shape: (B, F, N), onde F = floor(T/2) + 1
        Xf = torch.fft.rfft(x, dim=1)
        F = Xf.shape[1]

        # K = floor(T/2)
        K = F - 1

        # Pontos de corte
        c1 = K // 3
        c2 = (2 * K) // 3

        # Máscaras
        mask_low = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        mask_mid = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        mask_high = torch.zeros(F, device=x.device, dtype=Xf.dtype)

        # Bandas:
        # low  = {0, ..., c1}
        # mid  = {c1+1, ..., c2}
        # high = {c2+1, ..., K}
        mask_low[0:c1 + 1] = 1

        if c1 + 1 <= c2:
            mask_mid[c1 + 1:c2 + 1] = 1

        if c2 + 1 <= K:
            mask_high[c2 + 1:K + 1] = 1

        # Broadcast para (B, F, N)
        mask_low = mask_low.view(1, F, 1)
        mask_mid = mask_mid.view(1, F, 1)
        mask_high = mask_high.view(1, F, 1)

        # Aplicação das máscaras
        X_low = Xf * mask_low
        X_mid = Xf * mask_mid
        X_high = Xf * mask_high

        # Reconstrução temporal
        x_low = torch.fft.irfft(X_low, n=T, dim=1)
        x_mid = torch.fft.irfft(X_mid, n=T, dim=1)
        x_high = torch.fft.irfft(X_high, n=T, dim=1)

        return x_low, x_mid, x_high

class BandInstanceNorm(nn.Module):
    """
    Normalização por look-back:
    para cada amostra e para cada canal, ao longo do tempo.

    Entrada:
        x: (B, T, N)

    Saída:
        x_norm: (B, T, N)
        mean:   (B, 1, N)
        std:    (B, 1, N)
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)  # (B, 1, N)
        std = torch.sqrt(var + self.eps)

        x_norm = (x - mean) / std
        return x_norm, mean, std

    @staticmethod
    def denormalize(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        y_norm: (B, H, N) ou (B, H)
        mean:   (B, 1, N) ou compatível
        std:    (B, 1, N) ou compatível
        """
        return y_norm * std + mean

class BandExpertMLP(nn.Module):
    """
    Especialista simples por banda.

    Entrada:
        x: (B, T, N)

    Saída:
        y: (B, H, N)
    """

    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        # (B, T, N) -> (B, N, T)
        x = x.permute(0, 2, 1)

        # Aplica a MLP em cada canal
        # entrada da linear: último eixo = T
        y = self.net(x)  # (B, N, H)

        # (B, N, H) -> (B, H, N)
        y = y.permute(0, 2, 1)
        return y


class SumAggregator(nn.Module):
    """
    Agregação por soma simples:
        y = y_low + y_mid + y_high
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        return y_low + y_mid + y_high


class MLPAggregator(nn.Module):
    """
    Agregador via MLP no nível das previsões.

    Entrada:
        y_low, y_mid, y_high: (B, H, N)

    Saída:
        y: (B, H, N)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        # concatena no último eixo "de banda"
        # (B, H, N) -> (B, H, N, 1)
        y_low = y_low.unsqueeze(-1)
        y_mid = y_mid.unsqueeze(-1)
        y_high = y_high.unsqueeze(-1)

        # (B, H, N, 3)
        y_cat = torch.cat([y_low, y_mid, y_high], dim=-1)

        # aplica MLP em cada posição (b, h, n)
        y = self.mlp(y_cat).squeeze(-1)  # (B, H, N)
        return y                


class BandWiseForecastModel(nn.Module):
    """
    Modelo completo:
    - decomposição espectral por bandas
    - normalização por banda
    - especialista por banda
    - de-normalização por banda
    - agregação no nível das previsões

    Entrada:
        x: (B, T, N)

    Saída:
        y_hat: (B, H, N)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        expert_hidden_dim: int = 128,
        aggregator_type: str = "mlp",
        aggregator_hidden_dim: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decomposer = SpectralBandDecomposer()
        self.norm = BandInstanceNorm(eps=eps)

        self.low_expert = BandExpertMLP(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )
        self.mid_expert = BandExpertMLP(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )
        self.high_expert = BandExpertMLP(
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

        if aggregator_type == "sum":
            self.aggregator = SumAggregator()
        elif aggregator_type == "mlp":
            self.aggregator = MLPAggregator(
                hidden_dim=aggregator_hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"aggregator_type inválido: {aggregator_type}")

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        # 1) Decomposição
        x_low, x_mid, x_high = self.decomposer(x)

        # 2) Normalização por banda
        x_low_norm, low_mean, low_std = self.norm(x_low)
        x_mid_norm, mid_mean, mid_std = self.norm(x_mid)
        x_high_norm, high_mean, high_std = self.norm(x_high)

        # 3) Previsão por banda (normalizada)
        y_low_norm = self.low_expert(x_low_norm)    # (B, H, N)
        y_mid_norm = self.mid_expert(x_mid_norm)    # (B, H, N)
        y_high_norm = self.high_expert(x_high_norm) # (B, H, N)

        # 4) De-normalização por banda
        # low_mean, low_std: (B, 1, N), broadcast em H
        y_low = BandInstanceNorm.denormalize(y_low_norm, low_mean, low_std)
        y_mid = BandInstanceNorm.denormalize(y_mid_norm, mid_mean, mid_std)
        y_high = BandInstanceNorm.denormalize(y_high_norm, high_mean, high_std)

        # 5) Agregação final
        y_hat = self.aggregator(y_low, y_mid, y_high)

        return y_hat        