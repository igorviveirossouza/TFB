
import itertools
from typing import Dict, Optional, List

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
    "aggregator_type": "sum",   # "sum", "mlp" ou "pesos"
    "aggregator_hidden_dim": 64,
    "eps": 1e-5,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 30,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,

    "expert_type": "attention",   # "mlp" ou "attention"
    "expert_d_model": 16,
    "expert_n_heads": 4,
    "expert_ff_dim": 128,
    "channel_n_heads": 4,
    "temporal_pool_type": "attn",   # "avg", "max", "last", "attn", "flat"

    "n_bands": 3,
    "band_init": "uniform",              # "uniform" ou "log"
    "band_init_width": 0.18,
    "band_min_width": 1e-3,
    "normalize_band_masks": True,

    # auditoria
    "enable_band_audit": False,
    "print_band_audit": False,

    # Agregação:
    "channel_agg_type": "attention",   # "attention", "none", "linear", "residual_gated"
}


def _safe_mean_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlação média entre dois tensores com shape (B, T, N) ou (B, H, N).
    Calcula por (batch, canal) ao longo da dimensão temporal e depois tira média.
    Retorna escalar tensor.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shapes incompatíveis para correlação: {a.shape} vs {b.shape}")

    a0 = a - a.mean(dim=1, keepdim=True)
    b0 = b - b.mean(dim=1, keepdim=True)

    num = (a0 * b0).sum(dim=1)  # (B, N)
    den = torch.sqrt((a0.pow(2).sum(dim=1) + eps) * (b0.pow(2).sum(dim=1) + eps))  # (B, N)
    corr = num / den
    return corr.mean()


def _band_energy(x: torch.Tensor) -> torch.Tensor:
    """
    Energia média de um tensor (B, T, N) ou (B, H, N).
    Retorna escalar tensor.
    """
    return x.pow(2).mean()


def _pairwise_overlap_from_masks(masks: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    masks: (K, F)
    overlap cosseno entre pares de bandas.
    """
    out: Dict[str, torch.Tensor] = {}
    K = masks.shape[0]
    for i, j in itertools.combinations(range(K), 2):
        mi = masks[i]
        mj = masks[j]
        overlap = torch.dot(mi, mj) / (mi.norm() * mj.norm() + eps)
        out[f"overlap_{i}_{j}"] = overlap
    return out


def _pairwise_time_corr(prefix: str, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    xs: dict com tensores de shape (B, T, N) ou (B, H, N)
    """
    out: Dict[str, torch.Tensor] = {}
    keys = list(xs.keys())
    for a, b in itertools.combinations(keys, 2):
        out[f"{prefix}_corr_{a}_{b}"] = _safe_mean_corr(xs[a], xs[b])
    return out


def tensor_dict_to_python(d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Converte um dict de tensores escalares em floats Python para print/log.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            if v.ndim == 0:
                out[k] = float(v.detach().cpu())
            else:
                # fallback simples para vetores pequenos
                out[k] = float(v.detach().cpu().mean())
        else:
            out[k] = float(v)
    return out


def summarize_band_audits(audits: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """
    Resume uma lista de auditorias (um dict por batch) tomando a média por chave.
    """
    if len(audits) == 0:
        return {}

    keys = sorted({k for audit in audits for k in audit.keys()})
    summary: Dict[str, float] = {}

    for key in keys:
        vals = []
        for audit in audits:
            if key not in audit:
                continue
            v = audit[key]
            if torch.is_tensor(v):
                vals.append(float(v.detach().cpu().mean()))
            else:
                vals.append(float(v))
        if len(vals) > 0:
            summary[key] = sum(vals) / len(vals)

    return summary

class NoChannelAggregation(nn.Module):
    """
    Não faz agregação entre canais.
    Entrada e saída: (B, N, d)
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class LinearChannelMixer(nn.Module):
    """
    Mistura linear entre canais:
    para cada dimensão latente, aplica uma transformação linear no eixo dos canais.

    Entrada: (B, N, d)
    Saída:   (B, N, d)
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        B, N, d = z.shape

        # (B, N, d) -> (B, d, N)
        z_perm = z.transpose(1, 2)

        # peso compartilhado no eixo dos canais por batch
        # atenção: lazy parameterização simples não serve bem aqui,
        # então usamos uma projeção criada dinamicamente via einsum + parâmetro registrado.
        # Para ficar simples e robusto, criamos W no primeiro forward.
        if not hasattr(self, "W"):
            self.W = nn.Parameter(torch.eye(N, device=z.device, dtype=z.dtype))
            self.b = nn.Parameter(torch.zeros(N, device=z.device, dtype=z.dtype))

        mixed = torch.einsum("bdn,mn->bdm", z_perm, self.W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed)

        # volta para (B, N, d)
        return mixed.transpose(1, 2)


class ResidualGatedChannelMixer(nn.Module):
    """
    Mistura entre canais como correção residual gateada.

    z_out = z + gate(z) * mix(z)

    Entrada: (B, N, d)
    Saída:   (B, N, d)
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Esperado z com shape (B, N, d), recebido {z.shape}")

        B, N, d = z.shape
        z_perm = z.transpose(1, 2)  # (B, d, N)

        if not hasattr(self, "W"):
            self.W = nn.Parameter(torch.eye(N, device=z.device, dtype=z.dtype))
            self.b = nn.Parameter(torch.zeros(N, device=z.device, dtype=z.dtype))

        mixed = torch.einsum("bdn,mn->bdm", z_perm, self.W) + self.b.view(1, 1, N)
        mixed = self.dropout(mixed).transpose(1, 2)  # (B, N, d)

        g = self.gate(z)  # (B, N, d)
        return z + g * mixed



class LearnableSpectralBandDecomposer(nn.Module):
    """
    Separação espectral suave e aprendível.

    Mantém a interface do decomposer atual:
        x -> x_low, x_mid, x_high
    com cada saída tendo shape (B, T, N)

    A diferença é que as bandas agora são máscaras gaussianas aprendíveis.
    """

    def __init__(
        self,
        seq_len: int,
        n_bands: int = 3,
        init_mode: str = "uniform",
        init_width: float = 0.18,
        min_width: float = 1e-3,
        normalize_masks: bool = True,
    ):
        super().__init__()

        if n_bands != 3:
            raise ValueError("Esta versão foi feita para preservar low/mid/high, então n_bands deve ser 3.")

        self.seq_len = seq_len
        self.n_bands = n_bands
        self.freq_len = seq_len // 2 + 1
        self.min_width = min_width
        self.normalize_masks = normalize_masks

        if init_mode == "uniform":
            init_centers = torch.linspace(0.15, 0.85, steps=n_bands)
        elif init_mode == "log":
            # mais densidade em baixas frequências
            init_centers = torch.tensor([0.08, 0.28, 0.70], dtype=torch.float32)
        else:
            raise ValueError(f"band_init inválido: {init_mode}")

        init_widths = torch.full((n_bands,), init_width, dtype=torch.float32)

        self.raw_centers = nn.Parameter(self._inverse_sigmoid(init_centers))
        self.raw_widths = nn.Parameter(self._inverse_softplus(init_widths))

        freq_grid = torch.linspace(0.0, 1.0, steps=self.freq_len)
        self.register_buffer("freq_grid", freq_grid, persistent=False)

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(x))

    def _build_masks(self) -> torch.Tensor:
        """
        Retorna masks com shape (K, F)
        """
        centers = torch.sigmoid(self.raw_centers)  # (K,)
        widths = F.softplus(self.raw_widths) + self.min_width  # (K,)
        freq = self.freq_grid[None, :]  # (1, F)

        masks = torch.exp(-0.5 * ((freq - centers[:, None]) / widths[:, None]) ** 2)

        if self.normalize_masks:
            masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)

        return masks

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        _, T, _ = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        Xf = torch.fft.rfft(x.float(), dim=1)   # (B, F, N), complexo
        masks = self._build_masks()             # (3, F)

        outs = []
        for k in range(self.n_bands):
            mask_k = masks[k].view(1, self.freq_len, 1)   # (1, F, 1)
            Xk = Xf * mask_k
            xk = torch.fft.irfft(Xk, n=T, dim=1)          # (B, T, N)
            outs.append(xk)

        x_low, x_mid, x_high = outs
        return x_low, x_mid, x_high

    def regularization_loss(
        self,
        smoothness_weight: float = 1.0,
        diversity_weight: float = 1.0,
        coverage_weight: float = 1.0,
    ):
        masks = self._build_masks()  # (3, F)

        smooth = ((masks[:, 1:] - masks[:, :-1]) ** 2).mean()

        gram = masks @ masks.transpose(0, 1)
        eye = torch.eye(self.n_bands, device=gram.device, dtype=gram.dtype)
        diversity = ((gram * (1 - eye)) ** 2).mean()

        coverage = ((masks.sum(dim=0) - 1.0) ** 2).mean()

        total = (
            smoothness_weight * smooth
            + diversity_weight * diversity
            + coverage_weight * coverage
        )

        return {
            "smoothness": smooth,
            "diversity": diversity,
            "coverage": coverage,
            "total": total,
        }

    def band_params(self):
        centers = torch.sigmoid(self.raw_centers).detach()
        widths = (F.softplus(self.raw_widths) + self.min_width).detach()
        return {"centers": centers, "widths": widths}

    def audit(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Auditoria do decomposer.
        x: (B, T, N)

        Retorna métricas sobre máscaras e componentes decompostas.
        """
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        x_low, x_mid, x_high = self.forward(x)
        masks = self._build_masks()  # (3, F)

        audit: Dict[str, torch.Tensor] = {
            "mask_low_mean": masks[0].mean(),
            "mask_mid_mean": masks[1].mean(),
            "mask_high_mean": masks[2].mean(),
            "energy_x": _band_energy(x),
            "energy_low": _band_energy(x_low),
            "energy_mid": _band_energy(x_mid),
            "energy_high": _band_energy(x_high),
        }

        audit.update(_pairwise_overlap_from_masks(masks))
        audit.update(
            _pairwise_time_corr(
                "x_band",
                {"low": x_low, "mid": x_mid, "high": x_high},
            )
        )

        total_band_energy = audit["energy_low"] + audit["energy_mid"] + audit["energy_high"] + 1e-8
        audit["energy_ratio_low"] = audit["energy_low"] / total_band_energy
        audit["energy_ratio_mid"] = audit["energy_mid"] / total_band_energy
        audit["energy_ratio_high"] = audit["energy_high"] / total_band_energy

        params = self.band_params()
        centers = params["centers"]
        widths = params["widths"]
        audit["center_low"] = centers[0]
        audit["center_mid"] = centers[1]
        audit["center_high"] = centers[2]
        audit["width_low"] = widths[0]
        audit["width_mid"] = widths[1]
        audit["width_high"] = widths[2]

        return audit


class BandInstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        x_norm = (x - mean) / std
        return x_norm, mean, std

    @staticmethod
    def denormalize(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return y_norm * std + mean


class BandExpertMLP(nn.Module):
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

        _, T, _ = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x = x.permute(0, 2, 1)   # (B, N, T)
        y = self.net(x)          # (B, N, H)
        y = y.permute(0, 2, 1)   # (B, H, N)
        return y


class BandExpertAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_dim: int = 128,
        channel_n_heads: int = 4,
        channel_agg_type: str = "attention",
        temporal_pool_type: str = "avg",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.channel_agg_type = channel_agg_type

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} deve ser divisível por n_heads={n_heads}")
        if d_model % channel_n_heads != 0:
            raise ValueError(
                f"d_model={d_model} deve ser divisível por channel_n_heads={channel_n_heads}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

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

        self.temporal_pool_type = temporal_pool_type

        if self.temporal_pool_type == "avg":
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        elif self.temporal_pool_type == "max":
            self.temporal_pool = nn.AdaptiveMaxPool1d(1)
        elif self.temporal_pool_type == "last":
            self.temporal_pool = None
        elif self.temporal_pool_type == "attn":
            self.temporal_pool = None
            self.temporal_score = nn.Linear(d_model, 1)
        elif self.temporal_pool_type == "flat":
            self.temporal_pool = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(seq_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"temporal_pool_type inválido: {self.temporal_pool_type}")

        # -------- channel aggregation --------
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
            self.channel_block = ResidualGatedChannelMixer(
                d_model=d_model,
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

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, N)
        retorna: (B, pred_len, N)
        """
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        z = self.value_embedding(x) + self.pos_embedding  # (B*N, T, d)

        attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        z = self.temporal_norm1(z + attn_out)

        ffn_out = self.temporal_ffn(z)
        z = self.temporal_norm2(z + ffn_out)  # (B*N, T, d)

        if self.temporal_pool_type in ["avg", "max"]:
            z = z.transpose(1, 2)                 # (B*N, d, T)
            z = self.temporal_pool(z).squeeze(-1) # (B*N, d)
        elif self.temporal_pool_type == "last":
            z = z[:, -1, :]                       # (B*N, d)
        elif self.temporal_pool_type == "attn":
            alpha = torch.softmax(self.temporal_score(z), dim=1)  # (B*N, T, 1)
            z = torch.sum(alpha * z, dim=1)                       # (B*N, d)
        elif self.temporal_pool_type == "flat":
            z = self.temporal_pool(z)                             # (B*N, d)
        else:
            raise ValueError(f"temporal_pool_type inválido: {self.temporal_pool_type}")

        z = z.reshape(B, N, self.d_model)

       # c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
       #  c = self.channel_norm1(z + c_attn_out)

       #  c_ffn_out = self.channel_ffn(c)
       #  c = self.channel_norm2(c + c_ffn_out)  # (B, N, d)

    # 6) agregação entre canais
        if self.channel_agg_type == "attention":
            c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
            c = self.channel_norm1(z + c_attn_out)

            c_ffn_out = self.channel_ffn(c)
            c = self.channel_norm2(c + c_ffn_out)

        elif self.channel_agg_type == "none":
            c = self.channel_block(z)

        elif self.channel_agg_type in ["linear", "residual_gated"]:
            c = self.channel_block(z)
            c = self.channel_norm(c)

        else:
            raise ValueError(f"channel_agg_type inválido: {self.channel_agg_type}")


        y = self.head(c)
        y = y.permute(0, 2, 1).contiguous()
        return y


class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        return y_low + y_mid + y_high


class MLPAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        y_low = y_low.unsqueeze(-1)    # (B, H, N, 1)
        y_mid = y_mid.unsqueeze(-1)
        y_high = y_high.unsqueeze(-1)

        y_cat = torch.cat([y_low, y_mid, y_high], dim=-1)  # (B, H, N, 3)
        y = self.mlp(y_cat).squeeze(-1)                    # (B, H, N)
        return y


class LearnedLinearAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        normalized_weights = F.softmax(self.weights, dim=0)
        return 3 * (
            normalized_weights[0] * y_low
            + normalized_weights[1] * y_mid
            + normalized_weights[2] * y_high
        )

    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0).detach()


class BandWiseForecastModel(nn.Module):
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
        self.aggregator_type = configs.aggregator_type
        self.aggregator_hidden_dim = configs.aggregator_hidden_dim
        self.eps = configs.eps

        self.expert_type = configs.expert_type
        self.expert_d_model = configs.expert_d_model
        self.expert_n_heads = configs.expert_n_heads
        self.expert_ff_dim = configs.expert_ff_dim
        self.channel_n_heads = configs.channel_n_heads
        self.temporal_pool_type = configs.temporal_pool_type

        self.channel_agg_type = getattr(configs, "channel_agg_type", "attention")

        self.decomposer = LearnableSpectralBandDecomposer(
            seq_len=self.seq_len,
            n_bands=configs.n_bands,
            init_mode=configs.band_init,
            init_width=configs.band_init_width,
            min_width=configs.band_min_width,
            normalize_masks=configs.normalize_band_masks,
        )
        self.norm = BandInstanceNorm(eps=self.eps)

        if self.expert_type == "mlp":
            expert_cls = lambda: BandExpertMLP(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=self.expert_hidden_dim,
                dropout=self.dropout,
            )
        elif self.expert_type == "attention":
            expert_cls = lambda: BandExpertAttention(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                d_model=self.expert_d_model,
                n_heads=self.expert_n_heads,
                ff_dim=self.expert_ff_dim,
                channel_n_heads=self.channel_n_heads,
                channel_agg_type=self.channel_agg_type,
                temporal_pool_type=self.temporal_pool_type,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"expert_type inválido: {self.expert_type}")

        self.low_expert = expert_cls()
        self.mid_expert = expert_cls()
        self.high_expert = expert_cls()

        if self.aggregator_type == "sum":
            self.aggregator = SumAggregator()
        elif self.aggregator_type == "mlp":
            self.aggregator = MLPAggregator(
                hidden_dim=self.aggregator_hidden_dim,
                dropout=self.dropout,
            )
        elif self.aggregator_type == "pesos":
            self.aggregator = LearnedLinearAggregator()
        else:
            raise ValueError(f"aggregator_type inválido: {self.aggregator_type}")

    def audit_bands(
        self,
        x_enc: torch.Tensor,
        y_low: Optional[torch.Tensor] = None,
        y_mid: Optional[torch.Tensor] = None,
        y_high: Optional[torch.Tensor] = None,
        x_low: Optional[torch.Tensor] = None,
        x_mid: Optional[torch.Tensor] = None,
        x_high: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Consolida métricas de auditoria das bandas.
        """
        audit: Dict[str, torch.Tensor] = {}

        if hasattr(self.decomposer, "audit"):
            if x_low is None or x_mid is None or x_high is None:
                audit.update(self.decomposer.audit(x_enc))
            else:
                masks = self.decomposer._build_masks()
                audit["mask_low_mean"] = masks[0].mean()
                audit["mask_mid_mean"] = masks[1].mean()
                audit["mask_high_mean"] = masks[2].mean()

                audit["energy_x"] = _band_energy(x_enc)
                audit["energy_low"] = _band_energy(x_low)
                audit["energy_mid"] = _band_energy(x_mid)
                audit["energy_high"] = _band_energy(x_high)

                audit.update(_pairwise_overlap_from_masks(masks))
                audit.update(
                    _pairwise_time_corr(
                        "x_band",
                        {"low": x_low, "mid": x_mid, "high": x_high},
                    )
                )

                total_band_energy = audit["energy_low"] + audit["energy_mid"] + audit["energy_high"] + 1e-8
                audit["energy_ratio_low"] = audit["energy_low"] / total_band_energy
                audit["energy_ratio_mid"] = audit["energy_mid"] / total_band_energy
                audit["energy_ratio_high"] = audit["energy_high"] / total_band_energy

                params = self.decomposer.band_params()
                centers = params["centers"]
                widths = params["widths"]
                audit["center_low"] = centers[0]
                audit["center_mid"] = centers[1]
                audit["center_high"] = centers[2]
                audit["width_low"] = widths[0]
                audit["width_mid"] = widths[1]
                audit["width_high"] = widths[2]

        if y_low is not None and y_mid is not None and y_high is not None:
            audit["energy_y_low"] = _band_energy(y_low)
            audit["energy_y_mid"] = _band_energy(y_mid)
            audit["energy_y_high"] = _band_energy(y_high)

            audit.update(
                _pairwise_time_corr(
                    "y_band",
                    {"low": y_low, "mid": y_mid, "high": y_high},
                )
            )

        if hasattr(self.aggregator, "get_weights"):
            w = self.aggregator.get_weights()
            audit["agg_weight_low"] = w[0]
            audit["agg_weight_mid"] = w[1]
            audit["agg_weight_high"] = w[2]

        return audit

    def forecast(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        return_band_audit: bool = False,
    ):
        if x_enc.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x_enc.shape}")

        _, T, _ = x_enc.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x_low, x_mid, x_high = self.decomposer(x_enc)

        x_low_norm, low_mean, low_std = self.norm(x_low)
        x_mid_norm, mid_mean, mid_std = self.norm(x_mid)
        x_high_norm, high_mean, high_std = self.norm(x_high)

        y_low_norm = self.low_expert(x_low_norm)
        y_mid_norm = self.mid_expert(x_mid_norm)
        y_high_norm = self.high_expert(x_high_norm)

        y_low = self.norm.denormalize(y_low_norm, low_mean, low_std)
        y_mid = self.norm.denormalize(y_mid_norm, mid_mean, mid_std)
        y_high = self.norm.denormalize(y_high_norm, high_mean, high_std)

        dec_out = self.aggregator(y_low, y_mid, y_high)

        if torch.isnan(dec_out).any():
            print("WARNING: NaN detected in model output")

        if not return_band_audit:
            return dec_out

        band_audit = self.audit_bands(
            x_enc=x_enc,
            y_low=y_low,
            y_mid=y_mid,
            y_high=y_high,
            x_low=x_low,
            x_mid=x_mid,
            x_high=x_high,
        )

        return dec_out, band_audit

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        return_band_audit: bool = False,
    ):
        out = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            return_band_audit=return_band_audit,
        )

        if return_band_audit:
            dec_out, band_audit = out
            return dec_out[:, -self.pred_len:, :], band_audit

        dec_out = out
        return dec_out[:, -self.pred_len:, :]


class LearnableBandWiseAdapterAudit(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "BandWiseForecast"

    def _init_model(self):
        print("TIOMS FILE:", __file__, flush=True)
        print("expert_d_model:", self.config.expert_d_model, flush=True)
        print("expert_n_heads:", self.config.expert_n_heads, flush=True)
        print("expert_ff_dim:", self.config.expert_ff_dim, flush=True)
        print("channel_agg_type:", getattr(self.config, "channel_agg_type", "attention"), flush=True)
        return BandWiseForecastModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        print("MODEL DEVICE:", next(self.model.parameters()).device, flush=True)
        print("INPUT DEVICE:", input.device, flush=True)
        dec_input = target

        enable_band_audit = getattr(self.config, "enable_band_audit", False)
        print_band_audit = getattr(self.config, "print_band_audit", False)

        if enable_band_audit:
            output, band_audit = self.model(
                input,
                input_mark,
                dec_input,
                target_mark,
                return_band_audit=True,
            )

            if print_band_audit:
                print("BAND AUDIT:", tensor_dict_to_python(band_audit), flush=True)

            return {
                "output": output,
                "band_audit": band_audit,
            }

        output = self.model(
            input,
            input_mark,
            dec_input,
            target_mark
        )

        return {"output": output}

    @torch.no_grad()
    def collect_band_audit_batch(self, input, target, input_mark=None, target_mark=None) -> Dict[str, torch.Tensor]:
        """
        Coleta auditoria de um único batch.
        """
        self.model.eval()
        dec_input = target

        _, band_audit = self.model(
            input,
            input_mark,
            dec_input,
            target_mark,
            return_band_audit=True,
        )
        return band_audit

    @torch.no_grad()
    def collect_band_audit_from_loader(self, loader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Pipeline de auditoria:
        - percorre um loader
        - coleta auditoria por batch
        - resume tudo por média
        """
        self.model.eval()

        audits: List[Dict[str, torch.Tensor]] = []

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input = batch["input"]
                target = batch["target"]
                input_mark = batch.get("input_mark")
                target_mark = batch.get("target_mark")
            else:
                # fallback comum
                if len(batch) == 4:
                    input, target, input_mark, target_mark = batch
                elif len(batch) == 2:
                    input, target = batch
                    input_mark = None
                    target_mark = None
                else:
                    raise ValueError("Batch com formato não suportado para auditoria.")

            input = input.to(next(self.model.parameters()).device)
            target = target.to(next(self.model.parameters()).device)
            if input_mark is not None:
                input_mark = input_mark.to(next(self.model.parameters()).device)
            if target_mark is not None:
                target_mark = target_mark.to(next(self.model.parameters()).device)

            audit = self.collect_band_audit_batch(
                input=input,
                target=target,
                input_mark=input_mark,
                target_mark=target_mark,
            )
            audits.append(audit)

        return summarize_band_audits(audits)

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
