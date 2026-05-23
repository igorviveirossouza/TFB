# attention_factory.py

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


AttentionKind = Literal["mha", "paper_destationary", "custom_destationary"]
ConvMode = Literal["fixed", "learnable"]


class StandardMultiheadAttention(nn.Module):
    """
    Wrapper para manter a mesma interface das atenções customizadas.
    Ignora argumentos non-stationary.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        x_raw: Optional[torch.Tensor] = None,
        ns_covariates: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
        causal: bool = False,
        **kwargs,
    ):
        return self.attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )


class BaseCustomMultiheadAttention(nn.Module):
    """
    Multi-head attention implementada explicitamente para permitir:

        scores = tau * QK' + delta

    Assumimos batch_first=True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()

        if not batch_first:
            raise ValueError("As atenções customizadas assumem batch_first=True.")

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} deve ser divisível por num_heads={num_heads}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.eps = eps

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, E) -> (B, H, L, D)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # scores: (B, H, Lq, Lk)
        B, H, Lq, Lk = scores.shape

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                mask_value = torch.finfo(scores.dtype).min
                if attn_mask.dim() == 2:
                    # (Lq, Lk)
                    scores = scores.masked_fill(attn_mask[None, None, :, :], mask_value)
                elif attn_mask.dim() == 3:
                    # PyTorch MHA costuma usar (B*H, Lq, Lk)
                    if attn_mask.size(0) == B * H:
                        m = attn_mask.view(B, H, Lq, Lk)
                    elif attn_mask.size(0) == B:
                        m = attn_mask[:, None, :, :]
                    else:
                        raise ValueError(f"attn_mask 3D incompatível: {attn_mask.shape}")
                    scores = scores.masked_fill(m, mask_value)
                else:
                    raise ValueError(f"attn_mask deve ser 2D ou 3D, veio {attn_mask.shape}")
            else:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask[None, None, :, :].to(scores.dtype)
                elif attn_mask.dim() == 3:
                    if attn_mask.size(0) == B * H:
                        m = attn_mask.view(B, H, Lq, Lk)
                    elif attn_mask.size(0) == B:
                        m = attn_mask[:, None, :, :]
                    else:
                        raise ValueError(f"attn_mask 3D incompatível: {attn_mask.shape}")
                    scores = scores + m.to(scores.dtype)
                else:
                    raise ValueError(f"attn_mask deve ser 2D ou 3D, veio {attn_mask.shape}")

        if key_padding_mask is not None:
            # (B, Lk), True bloqueia
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], mask_value)

        return scores

    def _prepare_tau_delta(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        x_raw: Optional[torch.Tensor],
        ns_covariates: Optional[torch.Tensor],
        tau: Optional[torch.Tensor],
        delta: Optional[torch.Tensor],
        causal: bool,
    ):
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        x_raw: Optional[torch.Tensor] = None,
        ns_covariates: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
        causal: bool = False,
        **kwargs,
    ):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        q = self._shape_heads(self.q_proj(query))
        k = self._shape_heads(self.k_proj(key))
        v = self._shape_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        tau_hat, delta_hat = self._prepare_tau_delta(
            query=query,
            key=key,
            x_raw=x_raw,
            ns_covariates=ns_covariates,
            tau=tau,
            delta=delta,
            causal=causal,
        )

        # tau_hat:   (B, H, 1, 1) ou broadcastável
        # delta_hat: (B, H, 1, Lk) ou broadcastável
        scores = tau_hat * scores + delta_hat
        scores = self._apply_masks(scores, attn_mask, key_padding_mask)

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # (B, H, Lq, D)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            # média entre heads, compatível com nn.MultiheadAttention
            return out, attn.mean(dim=1)
        return out, None


class PaperDeStationaryAttention(BaseCustomMultiheadAttention):
    """
    Aproximação da De-stationary Attention do paper:

        log tau = MLP(std_x, x)
        delta   = MLP(mean_x, x)

        scores = tau * QK' + delta

    Aqui, x_raw deve estar em (B, Lk, C_raw). Se não for passado,
    usa key.detach() como aproximação.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        raw_dim: Optional[int] = None,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(embed_dim, num_heads, dropout, batch_first, **kwargs)
        self.raw_dim = raw_dim or embed_dim
        self.hidden_dim = hidden_dim

        in_dim = 2 * self.raw_dim
        self.tau_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def _raw_or_key(self, key: torch.Tensor, x_raw: Optional[torch.Tensor]) -> torch.Tensor:
        x = key if x_raw is None else x_raw
        if x.size(-1) != self.raw_dim:
            raise ValueError(
                f"raw_dim={self.raw_dim}, mas x_raw/key tem última dimensão {x.size(-1)}. "
                "Passe raw_dim corretamente no construtor."
            )
        return x

    def _prepare_tau_delta(
        self,
        query,
        key,
        x_raw,
        ns_covariates,
        tau,
        delta,
        causal,
    ):
        B, Lk, _ = key.shape

        if tau is not None:
            tau_hat = tau
        else:
            x = self._raw_or_key(key, x_raw)
            std = x.std(dim=1, unbiased=False)              # (B, C)
            last = x[:, -1, :]                              # (B, C)
            tau_in = torch.cat([std, last], dim=-1)          # (B, 2C)
            log_tau = self.tau_mlp(tau_in)                  # (B, H)
            tau_hat = torch.exp(log_tau).view(B, self.num_heads, 1, 1)

        if delta is not None:
            delta_hat = delta
        else:
            x = self._raw_or_key(key, x_raw)
            mean = x.mean(dim=1, keepdim=True).expand(-1, Lk, -1)  # (B, Lk, C)
            delta_in = torch.cat([mean, x], dim=-1)                # (B, Lk, 2C)
            delta_seq = self.delta_mlp(delta_in)                  # (B, Lk, H)
            delta_hat = delta_seq.permute(0, 2, 1).unsqueeze(2)    # (B,H,1,Lk)

        return tau_hat, delta_hat


class FixedOrLearnableDepthwiseConv1D(nn.Module):
    """
    Conv1D depthwise para suavização temporal.

    Entrada:  x (B, L, C)
    Saída:    y (B, L, C)

    mode="fixed": pesos = 1/kernel_size e congelados.
    mode="learnable": inicializa como média móvel e aprende.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        mode: ConvMode = "fixed",
        causal: bool = False,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size deve ser >= 1.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.mode = mode
        self.causal = causal

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

        with torch.no_grad():
            self.conv.weight.fill_(1.0 / kernel_size)

        if mode == "fixed":
            self.conv.weight.requires_grad_(False)
        elif mode == "learnable":
            self.conv.weight.requires_grad_(True)
        else:
            raise ValueError(f"mode inválido: {mode}")

    def forward(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        # x: (B, L, C)
        use_causal = self.causal if causal is None else causal
        B, L, C = x.shape
        if C != self.channels:
            raise ValueError(f"Esperado C={self.channels}, recebido C={C}.")

        xt = x.transpose(1, 2)  # (B,C,L)

        if use_causal:
            # padding só à esquerda; sem vazamento futuro
            pad_left = self.kernel_size - 1
            xt = F.pad(xt, (pad_left, 0), mode="replicate")
        else:
            # padding simétrico; se kernel par, mantém comprimento por corte posterior
            left = self.kernel_size // 2
            right = self.kernel_size - 1 - left
            xt = F.pad(xt, (left, right), mode="replicate")

        y = self.conv(xt)
        return y.transpose(1, 2)  # (B,L,C)


class CustomDeStationaryAttention(BaseCustomMultiheadAttention):
    """
    De-stationary attention personalizada:

    - tau vem de volatilidade local:
        r_t = x_t - x_{t-1}
        vol_t = Conv1D(r_t^2)

    - delta vem da tendência:
        trend_t = Conv1D(x_t)

    Ambas as convoluções podem ser fixas ou aprendíveis.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        raw_dim: Optional[int] = None,
        mean_kernel_size: int = 7,
        vol_kernel_size: int = 7,
        mean_conv_mode: ConvMode = "fixed",
        vol_conv_mode: ConvMode = "fixed",
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(embed_dim, num_heads, dropout, batch_first, **kwargs)

        self.raw_dim = raw_dim or embed_dim

        self.mean_conv = FixedOrLearnableDepthwiseConv1D(
            channels=self.raw_dim,
            kernel_size=mean_kernel_size,
            mode=mean_conv_mode,
        )
        self.vol_conv = FixedOrLearnableDepthwiseConv1D(
            channels=self.raw_dim,
            kernel_size=vol_kernel_size,
            mode=vol_conv_mode,
        )

        self.tau_proj = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

        self.delta_proj = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def _raw_or_key(self, key: torch.Tensor, x_raw: Optional[torch.Tensor]) -> torch.Tensor:
        x = key if x_raw is None else x_raw
        if x.size(-1) != self.raw_dim:
            raise ValueError(
                f"raw_dim={self.raw_dim}, mas x_raw/key tem última dimensão {x.size(-1)}."
            )
        return x

    @staticmethod
    def _returns(x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C)
        r = x[:, 1:, :] - x[:, :-1, :]
        first = torch.zeros_like(r[:, :1, :])
        return torch.cat([first, r], dim=1)

    def _prepare_tau_delta(
        self,
        query,
        key,
        x_raw,
        ns_covariates,
        tau,
        delta,
        causal,
    ):
        B, Lk, _ = key.shape
        x = self._raw_or_key(key, x_raw)

        if tau is not None:
            tau_hat = tau
        else:
            r = self._returns(x)
            rv = r.pow(2)
            vol = self.vol_conv(rv, causal=causal)          # (B,Lk,C)
            vol_summary = vol.mean(dim=1)                   # (B,C)
            tau_logits = self.tau_proj(vol_summary)         # (B,H)
            tau_hat = F.softplus(tau_logits).view(B, self.num_heads, 1, 1) + self.eps

        if delta is not None:
            delta_hat = delta
        else:
            trend = self.mean_conv(x, causal=causal)         # (B,Lk,C)
            delta_seq = self.delta_proj(trend)               # (B,Lk,H)
            delta_hat = delta_seq.permute(0, 2, 1).unsqueeze(2)  # (B,H,1,Lk)

        return tau_hat, delta_hat


def build_attention(
    attention_class: AttentionKind = "mha",
    embed_dim: int = 32,
    num_heads: int = 4,
    dropout: float = 0.1,
    batch_first: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory de atenção.

    attention_class:
        - "mha": MultiheadAttention clássico.
        - "paper_destationary": De-stationary Attention estilo paper.
        - "custom_destationary": tau por Conv1D em retornos² e delta por Conv1D em nível.
    """

    if attention_class == "mha":
        return StandardMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )

    if attention_class == "paper_destationary":
        return PaperDeStationaryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )

    if attention_class == "custom_destationary":
        return CustomDeStationaryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )

    raise ValueError(
        f"attention_class inválido: {attention_class}. "
        "Use 'mha', 'paper_destationary' ou 'custom_destationary'."
    )
