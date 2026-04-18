import torch
import torch.nn as nn
import torch.nn.functional as F
# =========================================================
# Embeddings
# =========================================================

class LinearScalarEmbedding(nn.Module):
    # z_t = W x_t + b
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NonlinearMultiFuncEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre = nn.Linear(1, hidden_dim)
        self.post = nn.Linear(3 * hidden_dim, d_model)
        self.eps = eps
        #self.mix_logits = nn.Parameter(torch.zeros(5))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)  # (M, T, H)
        #alpha = torch.softmax(self.mix_logits, dim=0)
        u = torch.cat(
            [
                F.gelu(h),
                torch.tanh(h),
                torch.sin(h),
                #alpha[3] * torch.pow(h, 2),
                #alpha[4] * torch.log(torch.abs(h) + self.eps),
            ],
            dim=-1,
        )
        

        return self.post(u)
        

class LagLinearEmbedding(nn.Module):
    # z_t = W [x_t, x_{t-1}, ..., x_{t-k+1}] + b
    # com padding causal à esquerda para preservar T
    def __init__(self, d_model: int, lag_size: int):
        super().__init__()
        self.lag_size = lag_size
        self.proj = nn.Linear(lag_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (M, T, 1)
        x1 = x.squeeze(-1)                         # (M, T)
        x_pad = F.pad(x1, (self.lag_size - 1, 0))
        windows = x_pad.unfold(dimension=1, size=self.lag_size, step=1)  # (M, T, lag)
        windows = windows.flip(-1)  # [x_t, x_{t-1}, ..., x_{t-k+1}]
        return self.proj(windows)


class Time2VecEmbedding(nn.Module):
    # t2v_t = [w0 t + b0, sin(w1 t + b1), ..., sin(wm t + bm)]
    def __init__(self, d_model: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, T, _ = x.shape
        device, dtype = x.device, x.dtype

        t = torch.arange(T, device=device, dtype=dtype).view(1, T, 1)
        linear = self.w0.to(dtype=dtype, device=device) * t + self.b0.to(dtype=dtype, device=device)
        periodic = torch.sin(
            t * self.w.to(dtype=dtype, device=device).view(1, 1, -1)
            + self.b.to(dtype=dtype, device=device).view(1, 1, -1)
        )
        z = torch.cat([linear, periodic], dim=-1)
        return z.expand(M, -1, -1)


class SpectralValueEmbedding(nn.Module):
    # s_t = [sin(nu_1 x_t), cos(nu_1 x_t), ..., sin(nu_m x_t), cos(nu_m x_t)]
    # z_t = P s_t
    def __init__(self, d_model: int, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = nn.Parameter(torch.randn(num_freqs))
        self.proj = nn.Linear(2 * num_freqs, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        phase = x * freqs
        feat = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.proj(feat)


class MixedEmbedding(nn.Module):
    # z_t = alpha_nl z_t^(nl) + alpha_t2v z_t^(t2v) + alpha_spec z_t^(spec)
    # alpha = softmax(a)
    def __init__(self, d_model: int, hidden_dim: int, num_freqs: int):
        super().__init__()
        self.nonlinear = NonlinearMultiFuncEmbedding(d_model, hidden_dim)
        self.time2vec = Time2VecEmbedding(d_model)
        self.spectral = SpectralValueEmbedding(d_model, num_freqs)
        self.mix_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_nl = self.nonlinear(x)
        z_t2v = self.time2vec(x)
        z_spec = self.spectral(x)

        alpha = torch.softmax(self.mix_logits, dim=0)
        return alpha[0] * z_nl + alpha[1] * z_t2v + alpha[2] * z_spec

    def get_mixing_weights(self) -> torch.Tensor:
        return torch.softmax(self.mix_logits, dim=0).detach().cpu()


def build_embedding(
    embedding_type: str,
    d_model: int,
    hidden_dim: int,
    lag_size: int,
    spectral_num_freqs: int,
) -> nn.Module:
    if embedding_type == "linear":
        return LinearScalarEmbedding(d_model)
    if embedding_type == "nonlinear":
        return NonlinearMultiFuncEmbedding(d_model, hidden_dim)
    if embedding_type == "lag_linear":
        return LagLinearEmbedding(d_model, lag_size)
    if embedding_type == "mixed":
        return MixedEmbedding(d_model, hidden_dim, spectral_num_freqs)
    if embedding_type == "spectral":
        return SpectralValueEmbedding(d_model, spectral_num_freqs)
    raise ValueError(f"embedding_type inválido: {embedding_type}")