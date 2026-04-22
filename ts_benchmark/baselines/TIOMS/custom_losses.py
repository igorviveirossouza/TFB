import torch
import torch.nn as nn


class TimeWeightedMSE(nn.Module):
    def __init__(self, K: int, decay_rate: float = 0.9):
        super().__init__()
        weights = torch.pow(
            torch.tensor(decay_rate, dtype=torch.float32),
            torch.arange(K, dtype=torch.float32),
        )
        weights = weights * (K / weights.sum())
        self.register_buffer("weights", weights.view(1, K, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred e target devem ter o mesmo shape. Recebi {pred.shape} e {target.shape}."
            )

        weights = self.weights.to(pred.device)
        if pred.ndim == 2:
            weights = weights.squeeze(-1)
        elif pred.ndim != 3:
            raise ValueError(
                f"TimeWeightedMSE espera tensores 2D ou 3D, mas recebeu {pred.ndim}D."
            )

        loss = (pred - target) ** 2
        weighted_loss = loss * weights
        return weighted_loss.mean()


class _SoftDTWBatch(nn.Module):
    """
    Implementação simples e totalmente em PyTorch de Soft-DTW para batch.
    Retorna um vetor (B,) com o valor Soft-DTW de cada amostra.
    """

    def __init__(self, gamma: float = 0.01):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma deve ser > 0, mas recebeu {gamma}.")
        self.gamma = float(gamma)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        if D.ndim != 3:
            raise ValueError(f"SoftDTW espera D com shape (B, T, T), mas recebeu {D.shape}.")

        B, N, M = D.shape
        inf = torch.tensor(float("inf"), device=D.device, dtype=D.dtype)
        R = torch.full((B, N + 2, M + 2), inf, device=D.device, dtype=D.dtype)
        R[:, 0, 0] = 0.0

        gamma = self.gamma
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                r0 = -R[:, i - 1, j - 1] / gamma
                r1 = -R[:, i - 1, j] / gamma
                r2 = -R[:, i, j - 1] / gamma
                rmax = torch.maximum(torch.maximum(r0, r1), r2)
                rsum = (
                    torch.exp(r0 - rmax)
                    + torch.exp(r1 - rmax)
                    + torch.exp(r2 - rmax)
                )
                softmin = -gamma * (torch.log(rsum) + rmax)
                R[:, i, j] = D[:, i - 1, j - 1] + softmin

        return R[:, N, M]


class DILATELoss(nn.Module):
    """
    Versão prática da DILATE para forecasting.

    Entrada aceita:
      - (B, H)
      - (B, H, 1)
      - (B, H, N)

    Para o caso multivariado, os canais são achatados para (B*N, H, 1)
    e a loss é calculada canal a canal.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.01):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha deve estar em [0,1], mas recebeu {alpha}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma deve ser > 0, mas recebeu {gamma}.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.soft_dtw = _SoftDTWBatch(gamma=gamma)

    @staticmethod
    def _reshape_series(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(-1)
        if x.ndim == 3:
            if x.shape[-1] == 1:
                return x
            b, h, n = x.shape
            return x.permute(0, 2, 1).contiguous().reshape(b * n, h, 1)
        raise ValueError(
            f"DILATELoss espera tensor com shape (B,H), (B,H,1) ou (B,H,N), mas recebeu {x.shape}."
        )

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, T, 1)
        return (x - y.transpose(1, 2)) ** 2

    def _soft_dtw_divergence(self, D_xy: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        D_xx = self._pairwise_sq_dist(x, x)
        D_yy = self._pairwise_sq_dist(y, y)
        return self.soft_dtw(D_xy) - 0.5 * self.soft_dtw(D_xx) - 0.5 * self.soft_dtw(D_yy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred e target devem ter o mesmo shape. Recebi {pred.shape} e {target.shape}."
            )

        pred_r = self._reshape_series(pred)
        target_r = self._reshape_series(target)

        B, T, _ = pred_r.shape
        D_xy = self._pairwise_sq_dist(pred_r, target_r)
        D_xy = D_xy.requires_grad_(True)

        shape_loss = self._soft_dtw_divergence(D_xy, pred_r, target_r)

        # Caminho/alinhamento suave: gradiente do soft-DTW em relação à matriz de custos.
        raw_sdtw = self.soft_dtw(D_xy)
        alignment = torch.autograd.grad(
            raw_sdtw.sum(),
            D_xy,
            create_graph=True,
            retain_graph=True,
        )[0]

        idx = torch.arange(T, device=pred.device, dtype=pred.dtype)
        omega = (idx[:, None] - idx[None, :]) ** 2
        omega = omega.unsqueeze(0).expand(B, T, T)
        temporal_loss = (alignment * omega).sum(dim=(1, 2)) / (T * T)

        total = self.alpha * shape_loss + (1.0 - self.alpha) * temporal_loss
        return total.mean()


def build_loss(
    loss_type: str,
    K: int,
    decay_rate: float = 0.9,
    alpha: float = 0.5,
    gamma: float = 0.01,
) -> nn.Module:
    if loss_type == "TimeWeightedMSE":
        criterion = TimeWeightedMSE(
            K=K,
            decay_rate=decay_rate,
        )
    elif loss_type == "DILATE":
        criterion = DILATELoss(
            alpha=alpha,
            gamma=gamma,
        )
    elif loss_type == "MAE":
        criterion = nn.L1Loss()
    elif loss_type == "Huber":
        criterion = nn.HuberLoss(delta=0.5)
    else:
        criterion = nn.MSELoss()

    return criterion