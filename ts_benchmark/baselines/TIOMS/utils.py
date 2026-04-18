import torch


def spectral_band_decompose(x: torch.Tensor):
    """
    Decompõe uma série temporal multivariada em três bandas espectrais
    (baixa, média, alta frequência), canal a canal.

    Parâmetros
    ----------
    x : torch.Tensor
        Tensor de entrada com shape (B, T, N),
        onde:
        B = batch size
        T = comprimento temporal (look-back)
        N = número de canais

    Retorna
    -------
    x_low : torch.Tensor
        Componente de baixa frequência, shape (B, T, N)
    x_mid : torch.Tensor
        Componente de média frequência, shape (B, T, N)
    x_high : torch.Tensor
        Componente de alta frequência, shape (B, T, N)
    """
    if x.ndim != 3:
        raise ValueError(f"Esperado tensor 3D (B, T, N), mas veio shape={x.shape}")

    B, T, N = x.shape

    # rFFT ao longo do tempo
    # shape: (B, F, N), com F = floor(T/2) + 1
    Xf = torch.fft.rfft(x, dim=1)
    F = Xf.shape[1]

    # K = floor(T/2), então F = K + 1
    K = F - 1

    # Pontos de corte
    c1 = K // 3
    c2 = (2 * K) // 3

    # Máscaras espectrais
    # shape: (F,)
    mask_low = torch.zeros(F, device=x.device, dtype=Xf.dtype)
    mask_mid = torch.zeros(F, device=x.device, dtype=Xf.dtype)
    mask_high = torch.zeros(F, device=x.device, dtype=Xf.dtype)

    mask_low[0:(c1 + 1)] = 1
    if c1 + 1 <= c2:
        mask_mid[(c1 + 1):(c2 + 1)] = 1
    if c2 + 1 <= K:
        mask_high[(c2 + 1):(K + 1)] = 1

    # Broadcast para (B, F, N)
    mask_low = mask_low.view(1, F, 1)
    mask_mid = mask_mid.view(1, F, 1)
    mask_high = mask_high.view(1, F, 1)

    # Filtragem espectral
    X_low = Xf * mask_low
    X_mid = Xf * mask_mid
    X_high = Xf * mask_high

    # Reconstrução temporal
    x_low = torch.fft.irfft(X_low, n=T, dim=1)
    x_mid = torch.fft.irfft(X_mid, n=T, dim=1)
    x_high = torch.fft.irfft(X_high, n=T, dim=1)

    return x_low, x_mid, x_high

class DilateLoss(nn.Module):
    """
    DILATE loss para forecasting multivariado.

    Para cada série univariada (por batch e por canal), calcula:
        L = alpha * L_shape + (1 - alpha) * L_time

    onde:
        - L_shape é Soft-DTW
        - L_time é a penalidade temporal baseada na matriz de alinhamento suave

    Entrada esperada:
        outputs, targets: (B, H, N)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(-1)
            targets = targets.unsqueeze(-1)

        if outputs.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, H, N), mas veio {outputs.shape}")

        B, H, N = outputs.shape
        total_loss = 0.0
        count = 0

        for b in range(B):
            for n in range(N):
                y_hat = outputs[b, :, n]   # (H,)
                y = targets[b, :, n]       # (H,)

                D = self._pairwise_distances(y_hat, y)               # (H, H)
                loss_shape, R = self._soft_dtw(D, self.gamma)        # escalar, matriz DP
                A = self._soft_alignment_path(D, R, self.gamma)      # (H, H)
                loss_time = self._temporal_loss(A)                   # escalar

                loss = self.alpha * loss_shape + (1.0 - self.alpha) * loss_time
                total_loss = total_loss + loss
                count += 1

        return total_loss / count

    @staticmethod
    def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (H,)
        retorna D_ij = (x_i - y_j)^2
        """
        return (x.unsqueeze(1) - y.unsqueeze(0)) ** 2

    @staticmethod
    def _softmin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
        vals = torch.stack([a, b, c], dim=0)
        return -gamma * torch.logsumexp(-vals / gamma, dim=0)

    def _soft_dtw(self, D: torch.Tensor, gamma: float):
        """
        Soft-DTW via programação dinâmica.
        D: (H, H)

        Retorna:
            value: escalar
            R: matriz DP de tamanho (H+2, H+2)
        """
        H = D.shape[0]
        inf = torch.tensor(float("inf"), device=D.device, dtype=D.dtype)

        R = torch.full((H + 2, H + 2), inf, device=D.device, dtype=D.dtype)
        R[0, 0] = 0.0

        for i in range(1, H + 1):
            for j in range(1, H + 1):
                r0 = R[i - 1, j]
                r1 = R[i, j - 1]
                r2 = R[i - 1, j - 1]
                soft = self._softmin(r0, r1, r2, gamma)
                R[i, j] = D[i - 1, j - 1] + soft

        return R[H, H], R

    def _soft_alignment_path(self, D: torch.Tensor, R: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Aproxima a matriz de alinhamento A do Soft-DTW.
        Implementação baseada no backward da DP.
        """
        H = D.shape[0]
        E = torch.zeros((H + 2, H + 2), device=D.device, dtype=D.dtype)
        E[H + 1, H + 1] = 1.0

        # padding sentinela para estabilidade
        D_pad = torch.zeros((H + 2, H + 2), device=D.device, dtype=D.dtype)
        D_pad[1:H + 1, 1:H + 1] = D

        for i in range(H, 0, -1):
            for j in range(H, 0, -1):
                a = torch.exp((R[i + 1, j] - R[i, j] - D_pad[i + 1, j]) / (-gamma))
                b = torch.exp((R[i, j + 1] - R[i, j] - D_pad[i, j + 1]) / (-gamma))
                c = torch.exp((R[i + 1, j + 1] - R[i, j] - D_pad[i + 1, j + 1]) / (-gamma))

                # proteção numérica
                a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
                c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

                E[i, j] = a * E[i + 1, j] + b * E[i, j + 1] + c * E[i + 1, j + 1]

        return E[1:H + 1, 1:H + 1]

    @staticmethod
    def _temporal_loss(A: torch.Tensor) -> torch.Tensor:
        """
        Penalidade temporal:
            sum_ij A_ij * ((i-j)^2) / H^2
        """
        H = A.shape[0]
        idx = torch.arange(H, device=A.device, dtype=A.dtype)
        Omega = (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2
        return (A * Omega).sum() / (H * H)
