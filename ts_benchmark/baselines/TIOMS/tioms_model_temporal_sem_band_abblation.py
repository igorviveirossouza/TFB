import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 24,
    "task_name": "long_term_forecast",
    "dropout": 0.1,
    "expert_hidden_dim": 128,
    "eps": 1e-5,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 30,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,

    # expert único
    "expert_type": "attention",   # "mlp" ou "attention"
    "expert_d_model": 16,
    "expert_n_heads": 4,
    "expert_ff_dim": 128,
    "channel_n_heads": 4,

    # agregação temporal dentro do expert de atenção
    # opções: "avg", "max", "last", "attn", "flat"
    "temporal_pool_type": "attn",

    # agregação entre canais
    # opções: "attention", "none", "linear", "residual_gated",
    #          "linear_residual", "linear_lowrank_residual", "mlp_mixer"
    "channel_agg_type": "attention",
    "channel_rank": 8,
    "channel_mlp_hidden_mult": 2,
}


class InstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, N)
        retorna:
            x_norm: (B, T, N)
            mean:   (B, 1, N)
            std:    (B, 1, N)
        """
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        x_norm = (x - mean) / std
        return x_norm, mean, std

    @staticmethod
    def denormalize(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        y_norm: (B, H, N)
        mean:   (B, 1, N)
        std:    (B, 1, N)
        """
        return y_norm * std + mean


class ExpertMLP(nn.Module):
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
        """
        x: (B, T, N)
        retorna: (B, pred_len, N)
        """
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        _, T, _ = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x = x.permute(0, 2, 1)   # (B, N, T)
        y = self.net(x)          # (B, N, pred_len)
        y = y.permute(0, 2, 1)   # (B, pred_len, N)
        return y


class NoChannelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class LinearChannelMixer(nn.Module):
    """
    Mistura linear pura no eixo dos canais:
        z_out = W z
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self._built = False

    def _build_if_needed(self, n_channels: int, device, dtype):
        if not self._built:
            self.W = nn.Parameter(torch.eye(n_channels, device=device, dtype=dtype))
            self.b = nn.Parameter(torch.zeros(n_channels, device=device, dtype=dtype))
            self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        _, N, _ = z.shape
        self._build_if_needed(N, z.device, z.dtype)

        z_perm = z.transpose(1, 2)  # (B, d, N)
        mixed = torch.einsum("bdn,mn->bdm", z_perm, self.W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed)
        return mixed.transpose(1, 2)


class ResidualGatedChannelMixer(nn.Module):
    """
    Mistura residual gateada:
        z_out = z + gate(z) * (W z)
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self._built = False

    def _build_if_needed(self, n_channels: int, device, dtype):
        if not self._built:
            self.W = nn.Parameter(torch.eye(n_channels, device=device, dtype=dtype))
            self.b = nn.Parameter(torch.zeros(n_channels, device=device, dtype=dtype))
            self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        _, N, _ = z.shape
        self._build_if_needed(N, z.device, z.dtype)

        z_perm = z.transpose(1, 2)
        mixed = torch.einsum("bdn,mn->bdm", z_perm, self.W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed).transpose(1, 2)

        g = self.gate(z)
        return z + g * mixed


class LinearResidualChannelMixer(nn.Module):
    """
    Mistura linear residual:
        z_out = z + W z
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self._built = False

    def _build_if_needed(self, n_channels: int, device, dtype):
        if not self._built:
            self.W = nn.Parameter(torch.eye(n_channels, device=device, dtype=dtype))
            self.b = nn.Parameter(torch.zeros(n_channels, device=device, dtype=dtype))
            self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        _, N, _ = z.shape
        self._build_if_needed(N, z.device, z.dtype)

        z_perm = z.transpose(1, 2)
        mixed = torch.einsum("bdn,mn->bdm", z_perm, self.W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed).transpose(1, 2)
        return z + mixed


class LowRankResidualChannelMixer(nn.Module):
    """
    Mistura low-rank residual:
        z_out = z + U V^T z
    """
    def __init__(self, rank: int = 8, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.dropout = nn.Dropout(dropout)
        self._built = False

    def _build_if_needed(self, n_channels: int, device, dtype):
        if not self._built:
            rank = min(self.rank, n_channels)
            self.U = nn.Parameter(0.02 * torch.randn(n_channels, rank, device=device, dtype=dtype))
            self.V = nn.Parameter(0.02 * torch.randn(n_channels, rank, device=device, dtype=dtype))
            self.b = nn.Parameter(torch.zeros(n_channels, device=device, dtype=dtype))
            self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        _, N, _ = z.shape
        self._build_if_needed(N, z.device, z.dtype)

        W = self.U @ self.V.T
        z_perm = z.transpose(1, 2)
        mixed = torch.einsum("bdn,mn->bdm", z_perm, W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed).transpose(1, 2)
        return z + mixed


class ChannelMLPMixer(nn.Module):
    """
    MLP Mixer no eixo dos canais:
        z_out = W2 GELU(W1 z)
    """
    def __init__(self, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_mult = hidden_mult
        self.dropout = nn.Dropout(dropout)
        self._built = False

    def _build_if_needed(self, n_channels: int, device, dtype):
        if not self._built:
            hidden = max(self.hidden_mult * n_channels, 1)
            self.fc1 = nn.Linear(n_channels, hidden, bias=True, device=device, dtype=dtype)
            self.fc2 = nn.Linear(hidden, n_channels, bias=True, device=device, dtype=dtype)
            self.act = nn.GELU()
            self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        _, N, _ = z.shape
        self._build_if_needed(N, z.device, z.dtype)

        z_perm = z.transpose(1, 2)
        mixed = self.fc2(self.dropout(self.act(self.fc1(z_perm))))
        mixed = self.dropout(mixed).transpose(1, 2)
        return mixed


class AttentiveTemporalPool(nn.Module):
    """
    Faz pooling atencional ao longo do tempo:
    entrada:  (B*, T, D)
    saída:    (B*, D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        # z: (B*, T, D)
        attn_logits = self.score(z)                  # (B*, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (z * attn_weights).sum(dim=1)       # (B*, D)
        return pooled


class ExpertAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_dim: int = 128,
        channel_n_heads: int = 4,
        channel_agg_type: str = "attention",
        channel_rank: int = 8,
        channel_mlp_hidden_mult: int = 2,
        dropout: float = 0.1,
        temporal_pool_type: str = "flat",
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} deve ser divisível por n_heads={n_heads}")
        if channel_agg_type == "attention" and d_model % channel_n_heads != 0:
            raise ValueError(
                f"d_model={d_model} deve ser divisível por channel_n_heads={channel_n_heads}"
            )

        valid_pool_types = {"avg", "max", "last", "attn", "flat"}
        if temporal_pool_type not in valid_pool_types:
            raise ValueError(
                f"temporal_pool_type inválido: {temporal_pool_type}. "
                f"Use um de {sorted(valid_pool_types)}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.temporal_pool_type = temporal_pool_type
        self.channel_agg_type = channel_agg_type

        self.value_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_norm2 = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        if self.temporal_pool_type == "attn":
            self.temporal_pool = AttentiveTemporalPool(d_model)
        elif self.temporal_pool_type == "flat":
            self.temporal_pool = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(seq_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.temporal_pool = None

        if self.channel_agg_type == "attention":
            self.channel_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=channel_n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.channel_norm1 = nn.LayerNorm(d_model)
            self.channel_norm2 = nn.LayerNorm(d_model)
            self.channel_ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(dropout),
            )
        elif self.channel_agg_type == "none":
            self.channel_block = NoChannelAggregation()
        elif self.channel_agg_type == "linear":
            self.channel_block = LinearChannelMixer(dropout=dropout)
            self.channel_norm = nn.LayerNorm(d_model)
        elif self.channel_agg_type == "residual_gated":
            self.channel_block = ResidualGatedChannelMixer(d_model=d_model, dropout=dropout)
            self.channel_norm = nn.LayerNorm(d_model)
        elif self.channel_agg_type == "linear_residual":
            self.channel_block = LinearResidualChannelMixer(dropout=dropout)
            self.channel_norm = nn.LayerNorm(d_model)
        elif self.channel_agg_type == "linear_lowrank_residual":
            self.channel_block = LowRankResidualChannelMixer(rank=channel_rank, dropout=dropout)
            self.channel_norm = nn.LayerNorm(d_model)
        elif self.channel_agg_type == "mlp_mixer":
            self.channel_block = ChannelMLPMixer(
                hidden_mult=channel_mlp_hidden_mult,
                dropout=dropout,
            )
            self.channel_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"channel_agg_type inválido: {self.channel_agg_type}")

        self.head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pred_len),
        )

    def _pool_time(self, z: torch.Tensor):
        if self.temporal_pool_type == "avg":
            return z.mean(dim=1)
        if self.temporal_pool_type == "max":
            return z.max(dim=1).values
        if self.temporal_pool_type == "last":
            return z[:, -1, :]
        if self.temporal_pool_type == "attn":
            return self.temporal_pool(z)
        if self.temporal_pool_type == "flat":
            return self.temporal_pool(z)
        raise RuntimeError(f"temporal_pool_type não tratado: {self.temporal_pool_type}")

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        z = self.value_embedding(x) + self.pos_embedding

        attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        z = self.temporal_norm1(z + attn_out)

        ffn_out = self.temporal_ffn(z)
        z = self.temporal_norm2(z + ffn_out)

        z = self._pool_time(z)
        z = z.reshape(B, N, self.d_model)

        if self.channel_agg_type == "attention":
            c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
            c = self.channel_norm1(z + c_attn_out)
            c_ffn_out = self.channel_ffn(c)
            c = self.channel_norm2(c + c_ffn_out)
        elif self.channel_agg_type == "none":
            c = self.channel_block(z)
        elif self.channel_agg_type in {
            "linear",
            "residual_gated",
            "linear_residual",
            "linear_lowrank_residual",
            "mlp_mixer",
        }:
            c = self.channel_block(z)
            c = self.channel_norm(c)
        else:
            raise ValueError(f"channel_agg_type inválido: {self.channel_agg_type}")

        y = self.head(c)
        y = y.permute(0, 2, 1).contiguous()
        return y


class NoBandWiseForecastModel(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.expert_hidden_dim = configs.expert_hidden_dim
        self.eps = configs.eps

        self.expert_type = configs.expert_type
        self.expert_d_model = configs.expert_d_model
        self.expert_n_heads = configs.expert_n_heads
        self.expert_ff_dim = configs.expert_ff_dim
        self.channel_n_heads = getattr(configs, "channel_n_heads", self.expert_n_heads)
        self.temporal_pool_type = getattr(configs, "temporal_pool_type", "attn")
        self.channel_agg_type = getattr(configs, "channel_agg_type", "attention")
        self.channel_rank = getattr(configs, "channel_rank", 8)
        self.channel_mlp_hidden_mult = getattr(configs, "channel_mlp_hidden_mult", 2)

        self.norm = InstanceNorm(eps=self.eps)

        if self.expert_type == "mlp":
            self.expert = ExpertMLP(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=self.expert_hidden_dim,
                dropout=self.dropout,
            )
        elif self.expert_type == "attention":
            self.expert = ExpertAttention(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                d_model=self.expert_d_model,
                n_heads=self.expert_n_heads,
                ff_dim=self.expert_ff_dim,
                channel_n_heads=self.channel_n_heads,
                channel_agg_type=self.channel_agg_type,
                channel_rank=self.channel_rank,
                channel_mlp_hidden_mult=self.channel_mlp_hidden_mult,
                dropout=self.dropout,
                temporal_pool_type=self.temporal_pool_type,
            )
        else:
            raise ValueError(f"expert_type inválido: {self.expert_type}")

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (B, T, N)
        retorna: (B, pred_len, N)
        """
        if x_enc.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x_enc.shape}")

        _, T, _ = x_enc.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        # normaliza por instância/canal ao longo do tempo
        x_norm, x_mean, x_std = self.norm(x_enc)

        # expert único sem bandas
        y_norm = self.expert(x_norm)

        # restaura escala original
        dec_out = self.norm.denormalize(y_norm, x_mean, x_std)

        if torch.isnan(dec_out).any():
            print("WARNING: NaN detected in model output", flush=True)

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class NoBandWiseAdapterChanel(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "NoBandWiseForecast"

    def _init_model(self):
        print("TIOMS FILE:", __file__, flush=True)
        print("expert_type:", self.config.expert_type, flush=True)
        print("expert_d_model:", self.config.expert_d_model, flush=True)
        print("expert_n_heads:", self.config.expert_n_heads, flush=True)
        print("expert_ff_dim:", self.config.expert_ff_dim, flush=True)
        print("channel_n_heads:", getattr(self.config, "channel_n_heads", self.config.expert_n_heads), flush=True)
        print("temporal_pool_type:", self.config.temporal_pool_type, flush=True)
        print("channel_agg_type:", getattr(self.config, "channel_agg_type", "attention"), flush=True)
        print("channel_rank:", getattr(self.config, "channel_rank", 8), flush=True)
        print("channel_mlp_hidden_mult:", getattr(self.config, "channel_mlp_hidden_mult", 2), flush=True)
        return NoBandWiseForecastModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        print("MODEL DEVICE:", next(self.model.parameters()).device, flush=True)
        print("INPUT DEVICE:", input.device, flush=True)

        dec_input = target

        output = self.model(
            input,
            input_mark,
            dec_input,
            target_mark
        )

        return {"output": output}

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ):
        return super().forecast_fit(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=1.0,
            **kwargs,
        )