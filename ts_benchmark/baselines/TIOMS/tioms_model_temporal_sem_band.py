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


class ExpertAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} deve ser divisível por n_heads={n_heads}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # embedding escalar -> vetor
        self.value_embedding = nn.Linear(1, d_model)

        # embedding posicional aprendível
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # atenção temporal por canal
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # cabeça final: concatena tempo x embedding e projeta para horizonte
        self.head = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, N)
        retorna: (B, pred_len, N)

        Observação:
        esta implementação faz atenção temporal separadamente em cada canal.
        Ou seja, não há mistura entre canais aqui.
        """
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        # (B, T, N) -> (B, N, T) -> (B*N, T, 1)
        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # embedding temporal
        z = self.value_embedding(x) + self.pos_embedding  # (B*N, T, d_model)

        # self-attention temporal
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm1(z + attn_out)

        # FFN
        ffn_out = self.ffn(z)
        z = self.norm2(z + ffn_out)

        # flatten temporal
        z = z.reshape(B * N, T * self.d_model)

        # projeção para horizonte
        y = self.head(z)  # (B*N, pred_len)

        # volta para (B, pred_len, N)
        y = y.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()
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
                dropout=self.dropout,
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


class NoBandWiseAdapter(DeepForecastingModelBase):
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