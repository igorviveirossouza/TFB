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