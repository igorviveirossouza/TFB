
import torch
import torch.nn as nn
from torch import optim

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
    "eps": 1e-5,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 30,
    "num_workers": 0,
    "loss": "TimeWeightedMSE",  # "MSE", "MAE", "Huber", "TimeWeightedMSE"
    "loss_decay_rate": 0.5,
    "patience": 3,

    # bloco temporal
    "d_model": 16,
    "n_heads": 4,
    "ff_dim": 128,

    # pooling temporal após atenção
    # opções: "avg", "max", "last", "attn", "flat"
    "temporal_pool_type": "attn",

    # agregação opcional entre canais, após o bloco temporal
    # opções: "none", "attention"
    "channel_agg_type": "attention",
    "channel_n_heads": 4,

    # Causal:
    # opções: "non_causal", "causal", "no_self"
    "causal_att": "non_causal",
}


class TimeWeightedMSE(nn.Module):
    """
    MSE ponderada no eixo temporal.

    Para horizonte K e decay_rate = lambda:
        w_t ∝ lambda^t,   t = 0, 1, ..., K-1

    Os pesos são normalizados para somarem K, mantendo escala próxima à MSE.
    """

    def __init__(self, K: int, decay_rate: float = 0.9):
        super().__init__()
        weights = torch.pow(torch.tensor(decay_rate, dtype=torch.float32), torch.arange(K, dtype=torch.float32))
        weights = weights * (K / weights.sum())
        self.register_buffer("weights", weights.view(1, K, 1))  # (1, K, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, K, N)
        weights = self.weights.to(pred.device)
        loss = (pred - target) ** 2
        weighted_loss = loss * weights
        return weighted_loss.mean()


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

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
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


class AttentiveTemporalPool(nn.Module):
    """
    Pooling atencional ao longo do tempo.
    entrada: (B*, T, D)
    saída:   (B*, D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        attn_logits = self.score(z)           # (B*, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (z * attn_weights).sum(dim=1)
        return pooled


class TemporalAttentionWithChannelAggregation(nn.Module):
    """
    1) Aplica atenção temporal separadamente em cada canal.
    2) Faz pooling temporal para obter um embedding por canal.
    3) Opcionalmente agrega entre canais.
    4) Decodifica cada canal para pred_len.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        temporal_pool_type: str = "attn",
        channel_agg_type: str = "attention",
        channel_n_heads: int = 4,
        causal_att: str = "non_causal",
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

        valid_channel_agg_types = {"none", "attention"}
        if channel_agg_type not in valid_channel_agg_types:
            raise ValueError(
                f"channel_agg_type inválido: {channel_agg_type}. "
                f"Use um de {sorted(valid_channel_agg_types)}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.temporal_pool_type = temporal_pool_type
        self.channel_agg_type = channel_agg_type
        self.causal_att = causal_att

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

        if temporal_pool_type == "attn":
            self.temporal_pool = AttentiveTemporalPool(d_model)
        elif temporal_pool_type == "flat":
            self.temporal_pool = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(seq_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.temporal_pool = None

        if channel_agg_type == "attention":
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
        if self.temporal_pool_type in {"attn", "flat"}:
            return self.temporal_pool(z)
        raise RuntimeError(f"temporal_pool_type não tratado: {self.temporal_pool_type}")

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

        # atenção temporal por canal
        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        z = self.value_embedding(x) + self.pos_embedding

        if self.causal_att == 'non_causal':
            attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        elif self.causal_att == 'causal':
            mask = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)
            attn_out, _ = self.temporal_attn(z, z, z, attn_mask=mask, need_weights=False)
        elif self.causal_att == 'no_self':
            mask = torch.eye(T, device=z.device, dtype=torch.bool)
            attn_out, _ = self.temporal_attn(z, z, z, attn_mask=mask, need_weights=False)
        else:
            raise ValueError(f"causal_att inválido: {self.causal_att}")

        z = self.temporal_norm1(z + attn_out)

        ffn_out = self.temporal_ffn(z)
        z = self.temporal_norm2(z + ffn_out)

        # um embedding por canal
        z = self._pool_time(z)                 # (B*N, D)
        z = z.reshape(B, N, self.d_model)      # (B, N, D)

        # agregação opcional entre canais
        if self.channel_agg_type == "attention":
            c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
            c = self.channel_norm1(z + c_attn_out)
            c_ffn_out = self.channel_ffn(c)
            c = self.channel_norm2(c + c_ffn_out)
        else:  # "none"
            c = z

        y = self.head(c)                       # (B, N, H)
        y = y.permute(0, 2, 1).contiguous()    # (B, H, N)
        return y


class AttentionForecastModel(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.eps = configs.eps

        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.ff_dim = configs.ff_dim
        self.temporal_pool_type = getattr(configs, "temporal_pool_type", "attn")
        self.channel_agg_type = getattr(configs, "channel_agg_type", "attention")
        self.channel_n_heads = getattr(configs, "channel_n_heads", self.n_heads)
        self.causal_att = getattr(configs, "causal_att", "non_causal")

        self.norm = InstanceNorm(eps=self.eps)

        self.block = TemporalAttentionWithChannelAggregation(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            temporal_pool_type=self.temporal_pool_type,
            channel_agg_type=self.channel_agg_type,
            channel_n_heads=self.channel_n_heads,
            causal_att=self.causal_att,
        )

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

        x_norm, x_mean, x_std = self.norm(x_enc)
        y_norm = self.block(x_norm)
        dec_out = self.norm.denormalize(y_norm, x_mean, x_std)

        if torch.isnan(dec_out).any():
            print("WARNING: NaN detected in model output", flush=True)

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class AttentionAdapterChannel(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "AttentionForecastModel"

    def _init_model(self):
        print("THIS FILE:", __file__, flush=True)
        print("d_model:", self.config.d_model, flush=True)
        print("n_heads:", self.config.n_heads, flush=True)
        print("ff_dim:", self.config.ff_dim, flush=True)
        print("temporal_pool_type:", self.config.temporal_pool_type, flush=True)
        print("channel_agg_type:", getattr(self.config, "channel_agg_type", "attention"), flush=True)
        print("channel_n_heads:", getattr(self.config, "channel_n_heads", self.config.n_heads), flush=True)
        print("loss:", getattr(self.config, "loss", "MSE"), flush=True)
        print("loss_decay_rate:", getattr(self.config, "loss_decay_rate", 0.9), flush=True)
        return AttentionForecastModel(self.config)

    def _init_criterion_and_optimizer(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.config.loss == "TimeWeightedMSE":
            criterion = TimeWeightedMSE(
                K=self.config.pred_len,
                decay_rate=getattr(self.config, "loss_decay_rate", 0.9),
            )
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        dec_input = target
        output = self.model(input, input_mark, dec_input, target_mark)
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
