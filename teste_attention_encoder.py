import torch
from ts_benchmark.baselines.TIOMS.attention_encoder import AttentionForecastModel

class Cfg:
    seq_len=36
    label_len=18
    pred_len=24
    enc_in=66
    c_out=66
    dropout=0.1
    eps=1e-5
    d_model=32
    n_heads=4
    ff_dim=128
    temporal_pool_type="last"
    channel_agg_type="none"
    channel_n_heads=4
    causal_att="no_self"
    embedding_type="nonlinear"
    embedding_hidden_dim=16
    lag_size=7
    spectral_num_freqs=18
    norm_type="revin"
    revin_affine=True
    aux_in=7
    aux_hidden_dim=32
    use_ohlcv_aux=True

model = AttentionForecastModel(Cfg())
x = torch.randn(2,36,66)
x_aux = torch.randn(2,36,66,7)
y = model(x, x_aux, None, None)
print(y.shape)
