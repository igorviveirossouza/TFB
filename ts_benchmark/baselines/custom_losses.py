import torch
import torch.nn as nn
import torch.nn.functional as F


POINTWISE_LOSSES = {"mse", "mae", "huber"}
RANKING_LOSSES = {
    "rank_hinge",
    "rank_margin",
    "rank_bpr",
    "ranknet",
    "whr1",
    "whr2",
    "listnet",
    "fingat",
}


def _cfg(config, name, default):
    return getattr(config, name, default)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower()
        if value in {"true", "1", "yes", "y", "sim", "s"}:
            return True
        if value in {"false", "0", "no", "n", "nao", "não"}:
            return False
    return bool(value)


def build_loss(config, normalizer_mean=None, normalizer_scale=None):
    """
    Factory central de losses para modelos deep do TFB.

    Parâmetros principais via model_hyper_params:
      loss: mse|mae|huber|rank_hinge|rank_margin|rank_bpr|ranknet|whr1|whr2|listnet|fingat
      loss_data_kind: log_return|simple_return|price
      loss_score_kind: log_return|simple_return
      loss_rank_lambda: peso do componente de ranking em losses combinadas
      loss_margin: margem das losses Hinge/Margin/WHR
      loss_ranknet_alpha: escala da RankNet
      loss_listnet_tau: temperatura da ListNet
      loss_fingat_delta: peso da BCE de movimento na FinGAT
      loss_inverse_norm: desfaz normalização do TFB dentro da loss antes de construir scores
    """
    loss_name = str(_cfg(config, "loss", "mse")).lower()

    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=float(_cfg(config, "huber_delta", 0.5)))

    if loss_name not in RANKING_LOSSES:
        raise ValueError(f"Loss desconhecida: {loss_name}")

    return FinancialRankingLoss(
        loss_name=loss_name,
        data_kind=str(_cfg(config, "loss_data_kind", "log_return")),
        score_kind=str(_cfg(config, "loss_score_kind", "log_return")),
        rank_lambda=float(_cfg(config, "loss_rank_lambda", 1.0)),
        margin=float(_cfg(config, "loss_margin", 0.0)),
        ranknet_alpha=float(_cfg(config, "loss_ranknet_alpha", 1.0)),
        listnet_tau=float(_cfg(config, "loss_listnet_tau", 1.0)),
        fingat_delta=float(_cfg(config, "loss_fingat_delta", 0.01)),
        inverse_norm=_as_bool(_cfg(config, "loss_inverse_norm", True)),
        normalizer_mean=normalizer_mean,
        normalizer_scale=normalizer_scale,
    )


def loss_accepts_base_value(criterion):
    return isinstance(criterion, FinancialRankingLoss) and criterion.needs_base_value


class FinancialRankingLoss(nn.Module):
    """
    Losses financeiras cross-section para saídas [B, H, N].

    A saída temporal do TFB é reduzida internamente para scores [B, N]:
      - log_return: soma dos log-retornos ao longo de H
      - simple_return: retorno simples acumulado ao longo de H
      - price: retorno entre preço-base t e preço previsto/observado em t+H

    Para dados de preço, base_value deve ser o último preço observado antes da janela futura.
    """

    def __init__(
        self,
        loss_name,
        data_kind="log_return",
        score_kind="log_return",
        rank_lambda=1.0,
        margin=0.0,
        ranknet_alpha=1.0,
        listnet_tau=1.0,
        fingat_delta=0.01,
        inverse_norm=True,
        normalizer_mean=None,
        normalizer_scale=None,
        eps=1e-8,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.data_kind = data_kind.lower()
        self.score_kind = score_kind.lower()
        self.rank_lambda = rank_lambda
        self.margin = margin
        self.ranknet_alpha = ranknet_alpha
        self.listnet_tau = listnet_tau
        self.fingat_delta = fingat_delta
        self.inverse_norm = inverse_norm
        self.eps = eps

        if normalizer_mean is not None:
            self.register_buffer(
                "normalizer_mean",
                torch.as_tensor(normalizer_mean, dtype=torch.float32).view(1, 1, -1),
            )
        else:
            self.normalizer_mean = None

        if normalizer_scale is not None:
            self.register_buffer(
                "normalizer_scale",
                torch.as_tensor(normalizer_scale, dtype=torch.float32).view(1, 1, -1),
            )
        else:
            self.normalizer_scale = None

    @property
    def needs_base_value(self):
        return self.data_kind in {"price", "prices", "preco", "preços"}

    def _maybe_inverse_norm(self, x):
        if not self.inverse_norm:
            return x
        if self.normalizer_mean is None or self.normalizer_scale is None:
            return x

        n = x.shape[-1]
        mean = self.normalizer_mean[..., :n].to(device=x.device, dtype=x.dtype)
        scale = self.normalizer_scale[..., :n].to(device=x.device, dtype=x.dtype)

        if x.dim() == 2:
            return x * scale.squeeze(1) + mean.squeeze(1)
        return x * scale + mean

    def _scores_from_series(self, pred, target, base_value=None):
        pred = self._maybe_inverse_norm(pred)
        target = self._maybe_inverse_norm(target)

        data_kind = self.data_kind
        score_kind = self.score_kind

        if data_kind in {"log_return", "log_returns", "log_retornos"}:
            pred_score = pred.sum(dim=1)
            target_score = target.sum(dim=1)

            if score_kind in {"simple_return", "simple_returns", "returns", "retornos"}:
                pred_score = torch.expm1(pred_score)
                target_score = torch.expm1(target_score)

            return pred_score, target_score

        if data_kind in {"simple_return", "simple_returns", "return", "returns", "retornos", "retornos_simples"}:
            pred_safe = torch.clamp(pred, min=-1.0 + self.eps)
            target_safe = torch.clamp(target, min=-1.0 + self.eps)

            if score_kind in {"log_return", "log_returns", "log_retornos"}:
                pred_score = torch.log1p(pred_safe).sum(dim=1)
                target_score = torch.log1p(target_safe).sum(dim=1)
            else:
                pred_score = torch.prod(1.0 + pred_safe, dim=1) - 1.0
                target_score = torch.prod(1.0 + target_safe, dim=1) - 1.0

            return pred_score, target_score

        if data_kind in {"price", "prices", "preco", "preços"}:
            if base_value is None:
                raise ValueError(
                    "Loss com loss_data_kind='price' exige base_value: último preço observado antes da janela futura."
                )

            base = self._maybe_inverse_norm(base_value)
            pred_last = pred[:, -1, :]
            target_last = target[:, -1, :]

            base = torch.clamp(base, min=self.eps)
            pred_ratio = torch.clamp(pred_last, min=self.eps) / base
            target_ratio = torch.clamp(target_last, min=self.eps) / base

            if score_kind in {"simple_return", "simple_returns", "returns", "retornos"}:
                return pred_ratio - 1.0, target_ratio - 1.0

            return torch.log(pred_ratio), torch.log(target_ratio)

        raise ValueError(f"loss_data_kind inválido: {self.data_kind}")

    def _pairwise_terms(self, pred_score, target_score):
        pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
        target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
        s_ij = torch.sign(target_diff)
        valid = s_ij != 0
        return pred_diff, target_diff, s_ij, valid

    def _masked_mean(self, values, mask):
        values = values[mask]
        if values.numel() == 0:
            return torch.zeros((), device=mask.device, dtype=torch.float32)
        return values.mean()

    def _hinge(self, pred_score, target_score, weighted=False, whr_mode="whr1"):
        pred_diff, _, s_ij, valid = self._pairwise_terms(pred_score, target_score)
        loss = F.relu(self.margin - s_ij * pred_diff)

        if weighted:
            weights = self._rank_weights(target_score, whr_mode)
            pair_weights = weights.unsqueeze(2) * weights.unsqueeze(1)
            loss = loss * pair_weights

        return self._masked_mean(loss, valid)

    def _bpr(self, pred_score, target_score):
        pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
        target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
        valid = target_diff > 0
        loss = F.softplus(-pred_diff)
        return self._masked_mean(loss, valid)

    def _ranknet(self, pred_score, target_score):
        pred_diff, _, s_ij, valid = self._pairwise_terms(pred_score, target_score)
        loss = F.softplus(-self.ranknet_alpha * s_ij * pred_diff)
        return self._masked_mean(loss, valid)

    def _listnet(self, pred_score, target_score):
        tau = max(self.listnet_tau, self.eps)
        p_true = F.softmax(target_score / tau, dim=1)
        log_p_pred = F.log_softmax(pred_score / tau, dim=1)
        return -(p_true * log_p_pred).sum(dim=1).mean()

    def _fingat_rank(self, pred_score, target_score):
        pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
        target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
        valid = target_diff != 0
        loss = F.relu(-(pred_diff * target_diff))
        return self._masked_mean(loss, valid)

    def _fingat_move(self, pred_score, target_score):
        target_move = (target_score > 0).to(dtype=pred_score.dtype)
        return F.binary_cross_entropy_with_logits(pred_score, target_move)

    def _rank_weights(self, target_score, mode):
        n = target_score.shape[1]
        order = torch.argsort(target_score, dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=target_score.dtype)
        rank_values = torch.arange(1, n + 1, device=target_score.device, dtype=target_score.dtype)
        ranks.scatter_(1, order, rank_values.unsqueeze(0).expand_as(ranks))

        if mode == "whr2":
            # peso exponencial decrescente: maior peso para o topo real
            denom = max(n - 1, 1)
            return torch.exp(-(ranks - 1.0) / denom)

        # WHR1: peso linear decrescente por posição no ranking real
        return (n - ranks + 1.0) / n

    def forward(self, pred, target, base_value=None):
        pred_score, target_score = self._scores_from_series(pred, target, base_value=base_value)

        if self.loss_name == "rank_hinge":
            rank_loss = self._hinge(pred_score, target_score)
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "rank_margin":
            rank_loss = self._hinge(pred_score, target_score)
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "rank_bpr":
            rank_loss = self._bpr(pred_score, target_score)
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "ranknet":
            rank_loss = self._ranknet(pred_score, target_score)
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "whr1":
            rank_loss = self._hinge(pred_score, target_score, weighted=True, whr_mode="whr1")
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "whr2":
            rank_loss = self._hinge(pred_score, target_score, weighted=True, whr_mode="whr2")
            point_loss = F.mse_loss(pred_score, target_score)
            return (1.0 - self.rank_lambda) * point_loss + self.rank_lambda * rank_loss

        if self.loss_name == "listnet":
            return self._listnet(pred_score, target_score)

        if self.loss_name == "fingat":
            rank_loss = self._fingat_rank(pred_score, target_score)
            move_loss = self._fingat_move(pred_score, target_score)
            return (1.0 - self.fingat_delta) * rank_loss + self.fingat_delta * move_loss

        raise ValueError(f"Loss desconhecida: {self.loss_name}")
