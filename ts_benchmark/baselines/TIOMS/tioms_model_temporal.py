import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

#expert_d_model = 32

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
    "aggregator_type": "sum",   # "sum", "mlp" ou "peso"
    "aggregator_hidden_dim": 64,
    "eps": 1e-5,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 30,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,

    # novos
    "expert_type": "attention",   # "mlp" ou "attention"
    "expert_d_model": 16, #expert_d_model,
    "expert_n_heads": 4, #max(expert_d_model // 4, 1),
    "expert_ff_dim": 128,
}

class SpectralBandDecomposer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        original_device = x.device
        x_cpu = x.detach().float().cpu().contiguous()

        _, T, _ = x_cpu.shape     

        #_, T, _ = x.shape
        
        Xf = torch.fft.rfft(x_cpu, dim=1)   # (B, F, N)
        F = Xf.shape[1]
        K = F - 1

        c1 = K // 3
        c2 = (2 * K) // 3

        # mask_low = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        # mask_mid = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        # mask_high = torch.zeros(F, device=x.device, dtype=Xf.dtype)
        
        mask_low  = torch.zeros(F, dtype=Xf.dtype)
        mask_mid  = torch.zeros(F, dtype=Xf.dtype)
        mask_high = torch.zeros(F,dtype=Xf.dtype)

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

        #return x_low, x_mid, x_high
        return (
            x_low.to(original_device),
            x_mid.to(original_device),
            x_high.to(original_device),
        )

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

## EXPERTS-----------------------------------------------------------------------------------------------

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

        # bloco de atenção temporal
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

        # cabeça de previsão
        self.head = nn.Linear(seq_len * d_model, pred_len)

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

## AGREGADORES ------------------------------------------------------------------------------------------        

class SumAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        return y_low + y_mid + y_high

class MLPAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim, bias = False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias = False),
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
        # Criamos 3 parâmetros treináveis, um para cada entrada
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, y_low: torch.Tensor, y_mid: torch.Tensor, y_high: torch.Tensor):
        # Aplicamos Softmax para que os pesos sejam positivos e somem 1.0
        # Isso transforma os parâmetros em coeficientes de importância (ex: 0.2, 0.5, 0.3)
        normalized_weights = F.softmax(self.weights, dim=0)
        
             
        return  3*(normalized_weights[0] * y_low + normalized_weights[1] * y_mid + normalized_weights[2] * y_high)

## ADAPTERS ------------------------------------------------------------------------------------------

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

        self.decomposer = SpectralBandDecomposer()
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
        elif self.aggregator_type == 'pesos':
             self.aggregator = LearnedLinearAggregator()   
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


class BandWiseAdapterTemp(DeepForecastingModelBase):
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
        return BandWiseForecastModel(self.config)

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