import warnings

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.time_series_library.layers.Conv_Blocks import Inception_Block_V1
from ts_benchmark.baselines.time_series_library.layers.Embed import DataEmbedding

warnings.filterwarnings("ignore")


def FFT_for_Period(x, k=2):

    B, T, C = x.shape

    xf = torch.fft.rfft(x, dim=1)
    F = xf.shape[1]

    # regra r = floor(T^(1/3))
    r = int(T ** (1/3))
    if r % 2 != 0:
        r -= 1
    r = max(r, 2)

    freq_indices = torch.arange(r, F, r, device=x.device)

    scores = []
    valid_freqs = []

    eps = 1e-5
    eye = torch.eye(C, device=x.device, dtype=torch.cfloat)

    for f in freq_indices:

        if f - r//2 < 0 or f + r//2 >= F:
            continue

        window = xf[:, f-r//2:f+r//2+1, :]

        # normalização energética
        window = window / (window.shape[1] ** 0.5)

        X = window

        S = torch.matmul(
            X.transpose(1,2).conj(),
            X
        ) / X.shape[1]

        # média no batch
        S_mean = S.mean(dim=0)

        # regularização
        S_mean = S_mean + eps * eye

        # autovalores da matriz espectral
        eigvals = torch.linalg.eigvalsh(S_mean).real

        if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
            continue

        lambda_max = eigvals.max()
        trace = eigvals.sum()

        if trace <= 0:
            continue

        score = lambda_max / trace

        scores.append(score)
        valid_freqs.append(f)

    # fallback se nenhuma frequência válida
    if len(scores) == 0:

        freqs = torch.tensor([1], device=x.device)

        periods = torch.tensor([max(3, T//2)], device=x.device)

        weights = torch.ones(B,1, device=x.device)

        return periods, weights, freqs


    scores = torch.stack(scores)

    k_eff = min(k, len(scores))

    _, idx = torch.topk(scores, k_eff, largest=True)

    freqs = torch.tensor(valid_freqs, device=x.device)[idx]

    periods = T // freqs

    periods = torch.clamp(periods, min=6, max=T//2)

    selected_scores = scores[idx]

    weights = torch.softmax(5 * selected_scores, dim=0)

    weights = weights.unsqueeze(0).repeat(B,1)

    return periods, weights, freqs


class TimesBlock(nn.Module):

    def __init__(self, configs):

        super(TimesBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model,
                configs.d_ff,
                num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff,
                configs.d_model,
                num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):

        B, T, N = x.size()

        period_list, period_weight, freqs = FFT_for_Period(x, self.k)

        print("DEBUG FFT RESULT")
        print("T:", x.shape[1])
        print("selected freqs:", freqs.detach().cpu())
        print("periods:", period_list.detach().cpu())

        res = []

        for i in range(period_list.shape[0]):

            period = int(period_list[i].item())

            if (self.seq_len + self.pred_len) % period != 0:

                length = (((self.seq_len + self.pred_len) // period) + 1) * period

                padding = torch.zeros(
                    [x.shape[0],
                     (length - (self.seq_len + self.pred_len)),
                     x.shape[2]]
                ).to(x.device)

                out = torch.cat([x, padding], dim=1)

            else:

                length = self.seq_len + self.pred_len
                out = x

            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            out = self.conv(out)

            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)

        period_weight = F.softmax(period_weight, dim=1)

        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        res = torch.sum(res * period_weight, -1)

        res = res + x

        return res


class MyTimesNet(nn.Module):

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

            self.predict_linear = nn.Linear(
                self.seq_len,
                self.pred_len + self.seq_len
            )

            self.projection = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True
            )

        if self.task_name == "imputation" or self.task_name == "anomaly_detection":

            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if self.task_name == "classification":

            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()

        x_enc = x_enc - means

        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )

        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out = self.predict_linear(
            enc_out.permute(0, 2, 1)
        ).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1
            )
        )

        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1
            )
        )

        if torch.isnan(dec_out).any():
            print("WARNING: NaN detected in model output")

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):

            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

            return dec_out[:, -self.pred_len:, :]

        return None