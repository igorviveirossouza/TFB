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

    
    "expert_type": "attention",   # "mlp" ou "attention"
    "expert_d_model": 16, #expert_d_model,
    "expert_n_heads": 4, #max(expert_d_model // 4, 1),
    "expert_ff_dim": 128,
    "channel_n_heads": 4,
    "temporal_pool_type": "attn",   # "avg", "max", "last", "attn", "flat"

    "n_bands": 3,
    "band_init": "uniform",              # "uniform" ou "log"
    "band_init_width": 0.18,
    "band_min_width": 1e-3,
    "normalize_band_masks": True,
}

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
            base = torch.tensor([0.08, 0.28, 0.70], dtype=torch.float32)
            init_centers = base
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

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        # FFT no device atual, sem detach/cpu
        Xf = torch.fft.rfft(x.float(), dim=1)   # (B, F, N), complexo
        masks = self._build_masks()             # (3, F)

        outs = []
        for k in range(self.n_bands):
            mask_k = masks[k].view(1, self.freq_len, 1)   # (1, F, 1)
            Xk = Xf * mask_k
            xk = torch.fft.irfft(Xk, n=T, dim=1)          # (B, T, N)
            outs.append(xk)

        # preserva interface atual
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
        channel_n_heads: int = 4,
        temporal_pool_type: str = "avg",
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} deve ser divisível por n_heads={n_heads}")
        if d_model % channel_n_heads != 0:
            raise ValueError(
                f"d_model={d_model} deve ser divisível por channel_n_heads={channel_n_heads}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # embedding escalar -> vetor
        self.value_embedding = nn.Linear(1, d_model)

        # embedding posicional temporal
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # -------- temporal self-attention --------
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

        # resumo temporal: L x d -> d
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
                nn.Flatten(start_dim=1),              # (B*N, T, d) -> (B*N, T*d)
                nn.Linear(seq_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        else:
            raise ValueError(f"temporal_pool_type inválido: {self.temporal_pool_type}")
        
        

        # -------- channel self-attention --------
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

        # head final por canal: d -> pred_len
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

        # -------------------------------------------------
        # 1) organizar por canal: (B, T, N) -> (B*N, T, 1)
        # -------------------------------------------------
        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # -------------------------------------------------
        # 2) embedding temporal
        # -------------------------------------------------
        z = self.value_embedding(x) + self.pos_embedding  # (B*N, T, d)

        # -------------------------------------------------
        # 3) temporal self-attention
        # -------------------------------------------------
        attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        z = self.temporal_norm1(z + attn_out)

        ffn_out = self.temporal_ffn(z)
        z = self.temporal_norm2(z + ffn_out)  # (B*N, T, d)

        # -------------------------------------------------
        # 4) resumir no tempo -> um vetor por canal
        #    (B*N, T, d) -> (B*N, d)
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 5) reorganizar por batch/canais: (B*N, d) -> (B, N, d)
        # -------------------------------------------------
        z = z.reshape(B, N, self.d_model)

        # -------------------------------------------------
        # 6) channel self-attention
        # -------------------------------------------------
        c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
        c = self.channel_norm1(z + c_attn_out)

        c_ffn_out = self.channel_ffn(c)
        c = self.channel_norm2(c + c_ffn_out)  # (B, N, d)

        # -------------------------------------------------
        # 7) head por canal: (B, N, d) -> (B, N, pred_len)
        # -------------------------------------------------
        y = self.head(c)

        # -------------------------------------------------
        # 8) voltar para (B, pred_len, N)
        # -------------------------------------------------
        y = y.permute(0, 2, 1).contiguous()
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
        self.channel_n_heads = configs.channel_n_heads

        self.decomposer = LearnableSpectralBandDecomposer(
            seq_len=self.seq_len,
            n_bands=configs.n_bands,
            init_mode=configs.band_init,
            init_width=configs.band_init_width,
            min_width=configs.band_min_width,
            normalize_masks=configs.normalize_band_masks,
        )
        self.norm = BandInstanceNorm(eps=self.eps)

        self.temporal_pool_type = configs.temporal_pool_type
        
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
                temporal_pool_type = self.temporal_pool_type,
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


class LearnableBandWiseAdapter(DeepForecastingModelBase):
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