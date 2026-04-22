import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.TIOMS.embeddings import build_embedding
from ts_benchmark.baselines.TIOMS.encoder import OHLCVFeatureEncoder
from ts_benchmark.baselines.TIOMS.custom_losses import build_loss



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
    "loss": "TimeWeightedMSE",  # "MSE", "MAE", "Huber", "TimeWeightedMSE", "DILATE"
    "loss_decay_rate": 0.5,
    "dilate_alpha": 0.5,
    "dilate_gamma": 0.01,
    "patience": 3,
    "d_model": 16,
    "n_heads": 4,
    "ff_dim": 128,
    "temporal_pool_type": "attn",
    "channel_agg_type": "attention",
    "channel_n_heads": 4,
    "causal_att": "non_causal",  # "non_causal", "causal", "no_self"

    # Embeddings
    "embedding_type": "linear",   # "linear", "nonlinear", "lag_linear", "mixed", "spectral"
    "embedding_hidden_dim": 32,
    "lag_size": 7,
    "spectral_num_freqs": 8,

    # Normalização
    "norm_type": "revin",     # "classic" | "revin"
    "revin_affine": True,

    # Encoder auxiliar OHLCV
    "aux_in": 7,
    "aux_hidden_dim": 32,
    "use_ohlcv_aux": True,
}



class InstanceNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        norm_type: str = "revin",
        revin_affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.norm_type = norm_type
        self.revin_affine = revin_affine
        self.affine_eps = 1e-8

        if self.norm_type == "revin" and self.revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        self.mean = None
        self.std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Esperado (B, T, N), veio {x.shape}")

        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x_norm = (x - self.mean) / self.std

        if self.norm_type == "revin" and self.revin_affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Chame normalize() antes de denormalize().")

        y = y_norm

        if self.norm_type == "revin" and self.revin_affine:
            y = (y - self.affine_bias) / (self.affine_weight + self.affine_eps)

        y = y * self.std + self.mean
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class AttentiveTemporalPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        attn_logits = self.score(z)
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (z * attn_weights).sum(dim=1)


class TemporalAttentionWithChannelAggregation(nn.Module):
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
        embedding_type: str = "linear",
        embedding_hidden_dim: int = 32,
        lag_size: int = 7,
        spectral_num_freqs: int = 8,
        aux_in: int = 7,
        aux_hidden_dim: int = 32,
        use_ohlcv_aux: bool = True,
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

        valid_embedding_types = {"linear", "nonlinear", "lag_linear", "mixed", "spectral"}
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"embedding_type inválido: {embedding_type}. "
                f"Use um de {sorted(valid_embedding_types)}"
            )

        valid_causal = {"non_causal", "causal", "no_self"}
        if causal_att not in valid_causal:
            raise ValueError(
                f"causal_att inválido: {causal_att}. "
                f"Use um de {sorted(valid_causal)}"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.temporal_pool_type = temporal_pool_type
        self.channel_agg_type = channel_agg_type
        self.causal_att = causal_att
        self.use_ohlcv_aux = use_ohlcv_aux

        self.embedding = build_embedding(
            embedding_type=embedding_type,
            d_model=d_model,
            hidden_dim=embedding_hidden_dim,
            lag_size=lag_size,
            spectral_num_freqs=spectral_num_freqs,
        )

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

        if self.use_ohlcv_aux:
            self.ohlcv_encoder = OHLCVFeatureEncoder(
                in_features=aux_in,
                d_model=d_model,
                hidden_dim=aux_hidden_dim,
                dropout=dropout,
                use_layernorm=True,
                use_residual=False,
            )
        else:
            self.ohlcv_encoder = None

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

    def forward(self, x: torch.Tensor, x_aux: torch.Tensor = None):
        if x.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x.shape}")

        B, T, N = x.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x_close = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        z = self.embedding(x_close)

        if self.use_ohlcv_aux:
            if x_aux is None:
                raise ValueError("x_aux é obrigatório quando use_ohlcv_aux=True")
            if x_aux.ndim != 4:
                raise ValueError(f"Esperado x_aux com shape (B,T,N,F), veio {x_aux.shape}")
            if x_aux.shape[:3] != (B, T, N):
                raise ValueError(
                    f"x_aux incompatível com x: esperado prefixo {(B,T,N)}, veio {x_aux.shape[:3]}"
                )

            f_aux = x_aux.shape[-1]
            x_aux = x_aux.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, f_aux)
            aux_z = self.ohlcv_encoder(x_aux)
            z = z + aux_z

        z = z + self.pos_embedding

        if self.causal_att == "non_causal":
            attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        elif self.causal_att == "causal":
            mask = torch.triu(
                torch.ones(T, T, device=z.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_out, _ = self.temporal_attn(z, z, z, attn_mask=mask, need_weights=False)
        elif self.causal_att == "no_self":
            mask = torch.eye(T, device=z.device, dtype=torch.bool)
            attn_out, _ = self.temporal_attn(z, z, z, attn_mask=mask, need_weights=False)
        else:
            raise RuntimeError(f"causal_att não tratado: {self.causal_att}")

        z = self.temporal_norm1(z + attn_out)
        z = self.temporal_norm2(z + self.temporal_ffn(z))

        z = self._pool_time(z)
        z = z.reshape(B, N, self.d_model)

        if self.channel_agg_type == "attention":
            c_attn_out, _ = self.channel_attn(z, z, z, need_weights=False)
            z = self.channel_norm1(z + c_attn_out)
            z = self.channel_norm2(z + self.channel_ffn(z))

        y = self.head(z)
        return y.permute(0, 2, 1).contiguous()


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

        self.embedding_type = getattr(configs, "embedding_type", "linear")
        self.embedding_hidden_dim = getattr(configs, "embedding_hidden_dim", 32)
        self.lag_size = getattr(configs, "lag_size", 7)
        self.spectral_num_freqs = getattr(configs, "spectral_num_freqs", 8)

        self.norm_type = getattr(configs, "norm_type", "classic")
        self.revin_affine = getattr(configs, "revin_affine", True)

        self.norm = InstanceNorm(
            num_features=self.enc_in,
            eps=self.eps,
            norm_type=self.norm_type,
            revin_affine=self.revin_affine,
        )

        self.aux_in = getattr(configs, "aux_in", 7)
        self.aux_hidden_dim = getattr(configs, "aux_hidden_dim", 32)
        self.use_ohlcv_aux = getattr(configs, "use_ohlcv_aux", True)

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
            embedding_type=self.embedding_type,
            embedding_hidden_dim=self.embedding_hidden_dim,
            lag_size=self.lag_size,
            spectral_num_freqs=self.spectral_num_freqs,
            aux_in=self.aux_in,
            aux_hidden_dim=self.aux_hidden_dim,
            use_ohlcv_aux=self.use_ohlcv_aux,
        )

    def forecast(self, x_enc, x_aux=None, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio {x_enc.shape}")

        _, T, _ = x_enc.shape
        if T != self.seq_len:
            raise ValueError(f"seq_len esperado={self.seq_len}, recebido={T}")

        x_norm = self.norm.normalize(x_enc)
        y_norm = self.block(x_norm, x_aux=x_aux)
        dec_out = self.norm.denormalize(y_norm)

        if torch.isnan(dec_out).any():
            print("WARNING: NaN detected in model output", flush=True)

        return dec_out

    def forward(
        self,
        x_enc,
        x_aux=None,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        dec_out = self.forecast(x_enc, x_aux, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


class AttentionAdapterChannelEnc(DeepForecastingModelBase):
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
        print("causal_att:", getattr(self.config, "causal_att", "non_causal"), flush=True)
        print("loss:", getattr(self.config, "loss", "MSE"), flush=True)
        print("loss_decay_rate:", getattr(self.config, "loss_decay_rate", 0.9), flush=True)
        print("dilate_alpha:", getattr(self.config, "dilate_alpha", 0.5), flush=True)
        print("dilate_gamma:", getattr(self.config, "dilate_gamma", 0.01), flush=True)
        print("embedding_type:", getattr(self.config, "embedding_type", "linear"), flush=True)
        print("embedding_hidden_dim:", getattr(self.config, "embedding_hidden_dim", 32), flush=True)
        print("lag_size:", getattr(self.config, "lag_size", 7), flush=True)
        print("spectral_num_freqs:", getattr(self.config, "spectral_num_freqs", 8), flush=True)
        print("pred_len:", self.config.pred_len, flush=True)
        print("seq_len:", self.config.seq_len, flush=True)

        model = AttentionForecastModel(self.config)

        if getattr(self.config, "embedding_type", "linear") == "mixed":
            weights = model.block.embedding.get_mixing_weights()
            print(
                f"mixed_embedding_init_weights: nl={weights[0]:.4f}, "
                f"t2v={weights[1]:.4f}, spec={weights[2]:.4f}",
                flush=True,
            )

        return model

    def _init_criterion_and_optimizer(self):
        criterion = build_loss(
            loss_type=getattr(self.config, "loss", "MSE"),
            K=self.config.pred_len,
            decay_rate=getattr(self.config, "loss_decay_rate", 0.9),
            alpha=getattr(self.config, "dilate_alpha", 0.5),
            gamma=getattr(self.config, "dilate_gamma", 0.01),
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    @staticmethod
    def _make_windows_2d(arr: np.ndarray, seq_len: int, pred_len: int):
        xs, ys = [], []
        total = len(arr)
        stop = total - seq_len - pred_len + 1
        if stop <= 0:
            raise ValueError(
                f"Série muito curta para seq_len={seq_len} e pred_len={pred_len}. "
                f"len={total}"
            )
        for i in range(stop):
            xs.append(arr[i : i + seq_len])
            ys.append(arr[i + seq_len : i + seq_len + pred_len])
        return np.stack(xs), np.stack(ys)

    @staticmethod
    def _make_windows_3d(arr: np.ndarray, seq_len: int, pred_len: int):
        xs = []
        total = len(arr)
        stop = total - seq_len - pred_len + 1
        if stop <= 0:
            raise ValueError(
                f"Série auxiliar muito curta para seq_len={seq_len} e pred_len={pred_len}. "
                f"len={total}"
            )
        for i in range(stop):
            xs.append(arr[i : i + seq_len])
        return np.stack(xs)

    def _train_loop(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_aux_train: np.ndarray = None,
    ):
        self.model.train()
        batch_size = int(self.config.batch_size)

        for _ in range(int(self.config.num_epochs)):
            perm = np.random.permutation(len(x_train))
            for start in range(0, len(x_train), batch_size):
                idx = perm[start : start + batch_size]

                xb = torch.tensor(x_train[idx], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y_train[idx], dtype=torch.float32, device=self.device)

                x_aux_b = None
                if x_aux_train is not None:
                    x_aux_b = torch.tensor(
                        x_aux_train[idx], dtype=torch.float32, device=self.device
                    )

                self.optimizer.zero_grad()
                out = self.model(
                    xb,
                    x_aux=x_aux_b,
                    x_mark_enc=None,
                    x_dec=yb,
                    x_mark_dec=None,
                )
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
    def _resolve_device(self):
        if hasattr(self, "device") and self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ):

        self.device = self._resolve_device()
        self.model = self._init_model().to(self.device)
        self.criterion, self.optimizer = self._init_criterion_and_optimizer()

        data = train_valid_data.to_numpy(dtype=np.float32)  # (T, N)

        aux_data = None
        if covariates is not None:
            aux_data = covariates.get("ohlcv_aux", None)
        if aux_data is not None:
            aux_data = np.asarray(aux_data, dtype=np.float32)  # (T, N, F)

        train_size = int(len(data) * float(train_ratio_in_tv))
        train_size = max(train_size, self.config.seq_len + self.config.pred_len)

        train_data = data[:train_size]
        x_train, y_train = self._make_windows_2d(
            train_data, self.config.seq_len, self.config.pred_len
        )

        x_aux_train = None
        if aux_data is not None:
            train_aux = aux_data[:train_size]
            x_aux_train = self._make_windows_3d(
                train_aux, self.config.seq_len, self.config.pred_len
            )

        self._train_loop(x_train, y_train, x_aux_train)
        return self

    def forecast(self, horizon, train, covariates=None):
        effective_horizon = int(horizon)
        if effective_horizon > int(self.config.pred_len):
            raise ValueError(
                f"horizon={effective_horizon} maior que pred_len={self.config.pred_len}"
            )

        self.model.eval()
        x = train.to_numpy(dtype=np.float32)

        if len(x) < self.config.seq_len:
            raise ValueError(
                f"Histórico curto demais: len={len(x)}, seq_len={self.config.seq_len}"
            )

        x = x[-self.config.seq_len :]
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)

        x_aux_b = None
        if covariates is not None and covariates.get("ohlcv_aux") is not None:
            aux = np.asarray(covariates["ohlcv_aux"], dtype=np.float32)
            aux = aux[-self.config.seq_len :]
            x_aux_b = torch.tensor(aux[None, ...], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(
                xb,
                x_aux=x_aux_b,
                x_mark_enc=None,
                x_dec=None,
                x_mark_dec=None,
            )

        return out[0, :effective_horizon].detach().cpu().numpy()

    def batch_forecast(self, horizon, batch_maker):
        effective_horizon = int(horizon)
        if effective_horizon > int(self.config.pred_len):
            raise ValueError(
                f"horizon={effective_horizon} maior que pred_len={self.config.pred_len}"
            )

        self.model.eval()
        all_predicts = []

        while batch_maker.has_more_batches():
            batch = batch_maker.make_batch(
                batch_size=int(self.config.batch_size),
                win_size=int(self.config.seq_len),
            )

            xb = torch.tensor(
                batch["input"], dtype=torch.float32, device=self.device
            )

            x_aux_b = None
            covariates = batch.get("covariates", None)
            if covariates is not None and covariates.get("ohlcv_aux") is not None:
                x_aux_b = torch.tensor(
                    covariates["ohlcv_aux"],
                    dtype=torch.float32,
                    device=self.device,
                )

            with torch.no_grad():
                out = self.model(
                    xb,
                    x_aux=x_aux_b,
                    x_mark_enc=None,
                    x_dec=None,
                    x_mark_dec=None,
                )

            all_predicts.append(out[:, :effective_horizon].detach().cpu().numpy())

        return np.concatenate(all_predicts, axis=0)