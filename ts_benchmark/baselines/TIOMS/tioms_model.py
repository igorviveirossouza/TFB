import torch
import torch.nn as nn

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
    "aggregator_type": "sum",   # "sum" ou "mlp"
    "aggregator_hidden_dim": 64,
    "eps": 1e-5,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
}


class SpectralBandDecomposer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        _, T, _ = x.shape

        Xf = torch.fft.rfft(x, dim=1)   # (B, F, N)
        F = Xf.shape[1]
        K = F - 1

        c1 = K // 3
        c2 = (2 * K) // 3

        mask_low = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        mask_mid = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        mask_high = torch.zeros(F, device=x.device, dtype=Xf.dtype)

        mask_low[0:c1 + 1] = 1
        if c1 + 1 <= c2:
            mask_mid[c1 + 1:c2 + 1] = 1
        if c2 + 1 <= K:
            mask_high[c2 + 1:K + 1] = 1

        mask_low = mask_low.view(1, F, 1)
        mask_mid = mask_mid.view(1, F, 1)
        mask_high = mask_high.view(1, F, 1)

        X_low = Xf * mask_low
        X_mid = Xf * mask_mid
        X_high = Xf * mask_high

        x_low = torch.fft.irfft(X_low, n=T, dim=1)
        x_mid = torch.fft.irfft(X_mid, n=T, dim=1)
        x_high = torch.fft.irfft(X_high, n=T, dim=1)

        return x_low, x_mid, x_high


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


class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        return y_low + y_mid + y_high


class MLPAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        y_low = y_low.unsqueeze(-1)    # (B, H, N, 1)
        y_mid = y_mid.unsqueeze(-1)
        y_high = y_high.unsqueeze(-1)

        y_cat = torch.cat([y_low, y_mid, y_high], dim=-1)  # (B, H, N, 3)
        y = self.mlp(y_cat).squeeze(-1)                    # (B, H, N)
        return y


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

        self.decomposer = SpectralBandDecomposer()
        self.norm = BandInstanceNorm(eps=self.eps)

        self.low_expert = BandExpertMLP(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
        )
        self.mid_expert = BandExpertMLP(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
        )
        self.high_expert = BandExpertMLP(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
        )

        if self.aggregator_type == "sum":
            self.aggregator = SumAggregator()
        elif self.aggregator_type == "mlp":
            self.aggregator = MLPAggregator(
                hidden_dim=self.aggregator_hidden_dim,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"aggregator_type inválido: {self.aggregator_type}")

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
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

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class BandWiseAdapter(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "BandWiseForecast"

    def _init_model(self):
        return BandWiseForecastModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
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