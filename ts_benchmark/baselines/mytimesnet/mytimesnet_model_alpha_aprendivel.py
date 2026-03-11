import warnings

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.time_series_library.layers.Conv_Blocks import Inception_Block_V1
from ts_benchmark.baselines.time_series_library.layers.Embed import DataEmbedding

warnings.filterwarnings("ignore")

def FFT_for_Period(x, k=2, alpha=0.5):

    B, T, C = x.shape

    xf = torch.fft.rfft(x, dim=1)
    F = xf.shape[1]

    # frequências disponíveis
    freqs = torch.arange(1, F, device=x.device).float()

    # escala espectral aprendível
    r = T ** alpha

    sigma = r / 2

    # kernel espectral contínuo
    weights_freq = torch.exp(-(freqs - r)**2 / (2 * sigma**2))

    weights_freq = weights_freq / weights_freq.sum()

    # aplica peso no espectro
    xf_band = xf[:,1:,:] * weights_freq.unsqueeze(0).unsqueeze(-1)

    # matriz espectral
    S = torch.matmul(
        xf_band.transpose(1,2).conj(),
        xf_band
    ) / xf_band.shape[1]

    S_mean = S.mean(dim=0)

    eps = 1e-6
    eye = torch.eye(C, device=x.device, dtype=torch.cfloat)

    S_mean = S_mean + eps * eye

    # determinante estável
    eigvals = torch.linalg.eigvalsh(S_mean).real
    eigvals = torch.clamp(eigvals, min=1e-8)

    logdet = torch.sum(torch.log(eigvals))

    # energia espectral
    power = torch.mean(torch.abs(xf)**2, dim=(0,2))

    power = power[1:] * weights_freq

    # top-k frequências
    k_eff = min(k, len(power))

    _, idx = torch.topk(power, k_eff)

    freqs_selected = idx + 1

    periods = T // freqs_selected

    periods = torch.clamp(periods, min=3, max=T//2)

    # pesos das convoluções
    weights = torch.softmax(power[idx], dim=0)

    weights = weights.unsqueeze(0).repeat(B,1)

    return periods, weights, freqs_selected

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):

        B, T, N = x.size()

        alpha = torch.sigmoid(self.alpha) * 0.6 + 0.2

        period_list, period_weight, freqs = FFT_for_Period(
            x,
            self.k,
            alpha
        )

        res = []

        for i in range(len(period_list)):

            period = int(period_list[i].item())

            if (self.seq_len + self.pred_len) % period != 0:

                length = (((self.seq_len + self.pred_len) // period) + 1) * period

                padding = torch.zeros(
                    [x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]],
                    device=x.device
                )

                out = torch.cat([x, padding], dim=1)

            else:

                length = self.seq_len + self.pred_len
                out = x

            out = (
                out.reshape(B, length // period, period, N)
                .permute(0,3,1,2)
                .contiguous()
            )

            out = self.conv(out)

            out = out.permute(0,2,3,1).reshape(B,-1,N)

            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)

        period_weight = F.softmax(period_weight, dim=1)

        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        res = torch.sum(res * period_weight, -1)

        res = res + x

        print("alpha:", alpha.detach().cpu().item())
        print("selected freqs:", freqs.detach().cpu())
        print("periods:", period_list.detach().cpu())

        return res

class MyTimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
       