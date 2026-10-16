"""Calculo vetorizado. Nenhuma leitura/escrita, rede ou dependencia de R/CUDA."""
from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class Strategy:
    id: str
    lado: str
    n_ativos: int
    filtrar_sinal: bool

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError("ID de estrategia invalido; use letras minusculas, numeros e _.")
        if self.lado not in {"compra", "venda"}:
            raise ValueError("lado deve ser compra ou venda.")
        if type(self.n_ativos) is not int or self.n_ativos < 1:
            raise ValueError("n_ativos deve ser inteiro positivo.")
        if type(self.filtrar_sinal) is not bool:
            raise ValueError("filtrar_sinal deve ser true ou false.")


def integers(values, label):
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all() or np.any(v != np.floor(v)):
        raise ValueError(f"{label}: indices nao inteiros/ausentes.")
    return v.astype(np.int64)


@dataclass
class Market:
    tickers: np.ndarray
    calendar: pd.DataFrame
    arrays: dict
    rf: np.ndarray
    received: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.calendar)

    @classmethod
    def from_frame(cls, frame, calendar, rf=None, received=None):
        numeric = ["abertura", "fechamento", "fechamento_execucao", "fechamento_anterior",
                   "fator_quantidade_evento", "provento_por_acao_com",
                   "valor_ativos_por_acao_com", "log_retorno_corrigido"]
        flags = ["pode_executar_abertura", "pode_executar_fechamento", "preco_preenchido"]
        required = ["date", "date_pregao", "data_anterior", "cols", *numeric, *flags]
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"Mercado: colunas ausentes: {sorted(missing)}")
        cal = calendar[["date", "date_pregao", "data_anterior"]].copy().sort_values("date")
        cal["date"] = integers(cal.date, "Calendario/date")
        for col in ("date_pregao", "data_anterior"):
            cal[col] = pd.to_datetime(cal[col], format="%Y-%m-%d", errors="raise")
        if (not len(cal) or not np.array_equal(cal.date, np.arange(1, len(cal) + 1))
                or cal.isna().any().any() or cal.date_pregao.duplicated().any()
                or not cal.date_pregao.is_monotonic_increasing
                or (cal.data_anterior >= cal.date_pregao).any()
                or not np.array_equal(cal.data_anterior.values[1:], cal.date_pregao.values[:-1])):
            raise ValueError("Calendario desalinhado; esperado date=1..N e pregoes consecutivos.")
        f = frame.copy()
        f["date"] = integers(f.date, "Mercado/date")
        if f.cols.isna().any() or not f.cols.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("Ticker ausente/invalido.")
        tickers = np.array(sorted(f.cols.unique()))
        if (not len(tickers) or f.duplicated(["date", "cols"]).any()
                or len(f) != len(cal) * len(tickers)):
            raise ValueError("Mercado duplicado ou nao balanceado.")
        f = f.sort_values(["date", "cols"]).reset_index(drop=True)
        if (not np.array_equal(f.date, np.repeat(cal.date, len(tickers)))
                or not np.array_equal(f.cols, np.tile(tickers, len(cal)))):
            raise ValueError("Chaves do mercado nao coincidem com o calendario/universo.")
        for col in ("date_pregao", "data_anterior"):
            dates = pd.to_datetime(f[col], format="%Y-%m-%d", errors="raise").values
            if not np.array_equal(dates, np.repeat(cal[col].values, len(tickers))):
                raise ValueError(f"Mercado/{col} desalinhado.")
        arrays = {c: pd.to_numeric(f[c], errors="raise").to_numpy(dtype=float).reshape(len(cal), -1)
                  for c in numeric}
        for c in flags:
            normalized = f[c].astype(str).str.lower().map({"true": True, "false": False})
            if normalized.isna().any():
                raise ValueError(f"Flag invalida: {c}")
            arrays[c] = normalized.to_numpy(dtype=bool).reshape(len(cal), -1)
        for c in numeric[1:]:
            if c == "fechamento_execucao":
                continue
            a = arrays[c]
            if not np.isfinite(a).all():
                raise ValueError(f"Valores nao finitos: {c}")
            if c in {"fechamento", "fechamento_anterior", "fator_quantidade_evento"} and np.any(a <= 0):
                raise ValueError(f"Valores nao positivos: {c}")
            if c in {"provento_por_acao_com", "valor_ativos_por_acao_com"} and np.any(a < 0):
                raise ValueError(f"Valores negativos: {c}")
        for quote, flag in [("abertura", "pode_executar_abertura"),
                            ("fechamento_execucao", "pode_executar_fechamento")]:
            valid = arrays[flag]
            if np.any(valid & (~np.isfinite(arrays[quote]) | (arrays[quote] <= 0)
                               | arrays["preco_preenchido"])):
                raise ValueError(f"Cotacao de execucao inconsistente: {quote}")
        c, p, q, d, v = (arrays[x] for x in ["fechamento", "fechamento_anterior",
            "fator_quantidade_evento", "provento_por_acao_com", "valor_ativos_por_acao_com"])
        if (not np.allclose(p[1:], c[:-1], rtol=1e-11, atol=1e-12)
                or not np.allclose(np.log((q*c+d+v)/p), arrays["log_retorno_corrigido"],
                                   rtol=1e-10, atol=1e-12)):
            raise ValueError("Eventos/fechamentos nao reproduzem os retornos corrigidos.")
        if not np.allclose(arrays["fechamento_execucao"][arrays["pode_executar_fechamento"]],
                           c[arrays["pode_executar_fechamento"]], rtol=1e-11, atol=1e-12):
            raise ValueError("Fechamento observado difere do fechamento de marcacao.")
        rates = np.full(len(cal), np.nan) if rf is None else np.asarray(rf, dtype=float)
        if rates.shape != (len(cal),) or np.isinf(rates).any() or np.any(rates <= -1):
            raise ValueError("Serie RF invalida.")
        market = cls(tickers, cal.reset_index(drop=True), arrays, rates)
        market.load_received(received, f)
        return market

    def load_received(self, frame, model_frame):
        """Trajetorias auxiliares por evento; nunca entram no universo do ranking.

        Quantidade e valor inicial devem reconciliar com V no painel congelado.
        A cobertura pode ser curta; possuir o ativo alem dela gera erro explicito.
        """
        if frame is None or frame.empty:
            return
        keys = ["id_distribuicao", "ticker_origem", "cols", "data_ex_distribuicao",
                "quantidade_recebida_por_acao_com", "inicio_negociacao_recebido",
                "fonte_evento_recebido", "fonte_cobertura_recebido"]
        values = ["fechamento", "fechamento_execucao", "fator_quantidade_evento",
                  "provento_por_acao_com"]
        needed = {*keys, *values, "date", "date_pregao", "pode_executar_fechamento", "preco_preenchido"}
        if not needed <= set(frame) or frame[list(needed - {"fechamento_execucao"})].isna().any().any():
            raise ValueError("Ativos recebidos: campos ausentes/incompletos.")
        totals, identifiers = {}, {}
        for event, group in frame.groupby("id_distribuicao", sort=True):
            if not isinstance(event, str) or not event or any(group[c].nunique() != 1 for c in keys):
                raise ValueError("Metadados inconsistentes do ativo recebido.")
            g = group.sort_values("date")
            meta = g.iloc[0]
            if meta.ticker_origem not in self.tickers or meta.cols == meta.ticker_origem:
                raise ValueError("Origem/especie invalida do ativo recebido.")
            ex = pd.to_datetime(meta.data_ex_distribuicao, format="%Y-%m-%d", errors="raise")
            start = pd.to_datetime(meta.inicio_negociacao_recebido, format="%Y-%m-%d", errors="raise")
            where = np.flatnonzero(self.calendar.date_pregao == ex)
            if len(where) != 1 or pd.isna(start) or start < ex:
                raise ValueError("Data ex/negociacao invalida do ativo recebido.")
            ex_index = int(where[0])
            steps = integers(g.date, "Recebido/date") - 1
            if (not np.array_equal(steps, np.arange(ex_index, ex_index+len(g)))
                    or steps[-1] >= self.n
                    or not np.array_equal(pd.to_datetime(g.date_pregao).values,
                                          self.calendar.date_pregao.values[steps])):
                raise ValueError(f"Trajetoria recebida incompleta/desalinhada: {event}")
            a = {c: pd.to_numeric(g[c], errors="raise").to_numpy(float) for c in values}
            for c in values:
                if c != "fechamento_execucao" and (not np.isfinite(a[c]).all() or
                        np.any(a[c] < 0 if c == "provento_por_acao_com" else a[c] <= 0)):
                    raise ValueError(f"Valor invalido em recebido/{c}: {event}")
            flags = {}
            for c in ("pode_executar_fechamento", "preco_preenchido"):
                flags[c] = g[c].astype(str).str.lower().map({"true": True, "false": False})
                if flags[c].isna().any():
                    raise ValueError(f"Flag invalida do ativo recebido: {event}")
                flags[c] = flags[c].to_numpy(bool)
            ok = flags["pode_executar_fechamento"]
            if (np.any(ok & (flags["preco_preenchido"] | ~np.isfinite(a["fechamento_execucao"])))
                    or not np.allclose(a["fechamento_execucao"][ok], a["fechamento"][ok], rtol=1e-11, atol=1e-12)
                    or np.any(ok & (self.calendar.date_pregao.values[steps] < start.to_datetime64()))):
                raise ValueError(f"Execucao inconsistente do ativo recebido: {event}")
            if a["fator_quantidade_evento"][0] != 1 or a["provento_por_acao_com"][0] != 0:
                raise ValueError("Entrega deve estar em unidades ex, sem provento duplicado na data ex.")
            ratio = float(meta.quantidade_recebida_por_acao_com)
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Proporcao invalida do ativo recebido.")
            parent = int(np.flatnonzero(self.tickers == meta.ticker_origem)[0])
            key = (ex_index, parent)
            scale = np.cumprod(a["fator_quantidade_evento"])
            cash = np.cumsum(a["provento_por_acao_com"] * np.r_[1., scale[:-1]])
            self.received.setdefault(key, []).append({"id": event, "ticker": meta.cols,
                "ratio": ratio, "close": a["fechamento"], "exit": a["fechamento_execucao"],
                "exit_ok": ok, "scale": scale, "cash": cash})
            totals[key] = totals.get(key, 0.) + ratio*a["fechamento"][0]
            identifiers.setdefault(key, set()).add(event)
        for (ex, parent), total in totals.items():
            if not np.isclose(total, self.arrays["valor_ativos_por_acao_com"][ex, parent], rtol=1e-10, atol=1e-12):
                raise ValueError("Ativo recebido nao reproduz o valor da distribuicao no painel congelado.")
            row = model_frame.iloc[ex*len(self.tickers)+parent]
            ids = row.get("ids_distribuicoes")
            if not isinstance(ids, str) or set(ids.split(";")) != identifiers[(ex, parent)]:
                raise ValueError("IDs de distribuicoes recebidas divergem da auditoria do modelo.")


def cumulative_targets(market, origins, k):
    # origin_step e um indice de data 1-based: a primeira linha-alvo e arrays[origin].
    days = np.asarray(origins)[:, None] + np.arange(k)
    return market.arrays["log_retorno_corrigido"][days].sum(axis=1)


def spearman_rows(predicted, actual):
    """Equivalente a cor(method='spearman'): pares finitos, empates medios, n>=3."""
    x, y = np.asarray(predicted, float), np.asarray(actual, float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("IC requer matrizes compativeis origens x ativos.")
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    rx = rankdata(np.where(valid, x, np.nan), axis=1, method="average", nan_policy="omit")
    ry = rankdata(np.where(valid, y, np.nan), axis=1, method="average", nan_policy="omit")
    # A media dos ranks e (n+1)/2, inclusive com empates.
    dx = np.where(valid, rx - (n[:, None]+1)/2, 0.)
    dy = np.where(valid, ry - (n[:, None]+1)/2, 0.)
    den = np.sqrt((dx*dx).sum(axis=1) * (dy*dy).sum(axis=1))
    good = (n >= 3) & (den > 0)
    ic = np.full(len(x), np.nan)
    np.divide((dx*dy).sum(axis=1), den, out=ic, where=good)
    status = np.where(n < 3, "MENOS_DE_3_PARES", np.where(den == 0, "RANK_CONSTANTE", "OK"))
    return np.clip(ic, -1, 1), n, status


def ic_frame(scores, origins, k, market, common_last_origin):
    """IC acumulado em K, mantido para reproduzir a agregacao do R antigo."""
    ic, n, status = spearman_rows(scores, cumulative_targets(market, origins, k))
    return pd.DataFrame({"k": k, "origin_step": origins, "target_step": origins+k,
                         "data_origem": market.calendar.date_pregao.values[origins-1],
                         "data_alvo": market.calendar.date_pregao.values[origins+k-1],
                         "spearman": ic, "n_pares": n, "status_ic": status,
                         "amostra_comum": origins <= common_last_origin})


def ic_daily_frame(predicted, origins, market, common_last_origin):
    """IC natural: previsao diaria para T contra retorno corrigido realizado em T.

    Cada par (origem, h) e uma cross-section independente. Nao mistura previsoes
    feitas em diferentes origens para uma mesma data-alvo.
    """
    count, horizon, assets = predicted.shape
    target_indices = origins[:, None] + np.arange(horizon)
    actual = market.arrays["log_retorno_corrigido"][target_indices]
    ic, n, status = spearman_rows(predicted.reshape(-1, assets), actual.reshape(-1, assets))
    return pd.DataFrame({"h": np.tile(np.arange(1, horizon+1), count),
                         "origin_step": np.repeat(origins, horizon),
                         "target_step": target_indices.ravel()+1,
                         "data_origem": np.repeat(market.calendar.date_pregao.values[origins-1], horizon),
                         "data_alvo": market.calendar.date_pregao.values[target_indices.ravel()],
                         "spearman": ic, "n_pares": n, "status_ic": status,
                         "amostra_comum": np.repeat(origins <= common_last_origin, horizon)})


def ic_summary(ic, dimension="k"):
    rows = []
    for k, frame in ic.groupby(dimension, sort=True):
        for sample in ("completa", "comum"):
            f = frame if sample == "completa" else frame[frame.amostra_comum]
            v = f.spearman.dropna()
            sd = v.std(ddof=1)
            rows.append({dimension: k, "amostra": sample, "n_origens": len(f), "n_ic_validos": len(v),
                         "ic_medio": v.mean(), "ic_mediano": v.median(), "ic_desvio": sd,
                         "fracao_ic_positivo": (v > 0).mean() if len(v) else np.nan,
                         "ic_ir": v.mean()/sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def metrics(daily, periods):
    r = daily.retorno.to_numpy()
    rf = daily.rf.to_numpy()
    eq = daily.patrimonio.to_numpy()
    sd = np.std(r, ddof=1) if len(r) > 1 else np.nan
    vol = sd*np.sqrt(252)
    cagr = np.expm1(np.log(eq[-1])*252/len(r))
    has_rf = np.isfinite(rf).all()
    ex = r-rf
    sdex = np.std(ex, ddof=1) if has_rf and len(r) > 1 else np.nan
    rfa = np.expm1(np.log1p(rf).sum()*252/len(r)) if has_rf else np.nan
    return pd.DataFrame([{
        "inicio": daily.data.iloc[0], "fim": daily.data.iloc[-1], "n_pregoes": len(r),
        "n_periodos": len(periods), "media_n_ativos": periods.n_ativos.mean(),
        "retorno_acumulado": eq[-1]-1, "retorno_anualizado": cagr,
        "desvio_padrao_diario": sd, "volatilidade_anualizada": vol,
        "sharpe_rf_zero": np.mean(r)/sd*np.sqrt(252) if sd > 0 else np.nan,
        "sharpe_cdi": np.mean(ex)/sdex*np.sqrt(252) if sdex > 0 else np.nan,
        "sharpe_geometrico_cdi": (cagr-rfa)/vol if has_rf and vol > 0 else np.nan,
        "status_rf": "BCB_SGS12" if has_rf else "NAO_FORNECIDA",
        "max_drawdown": daily.drawdown.min(),
        "cdi_acumulado": np.expm1(np.log1p(rf).sum()) if has_rf else np.nan,
        "cdi_anualizado": rfa,
        "fracao_blocos_caixa": (periods.n_ativos == 0).mean(),
    }])


def backtest(scores, origins, k, market, strategy, slippage_bps=0., cost_bps=0.):
    """Todos os blocos de uma estrategia de uma vez; quantidades fixas no bloco.

    Entrada em O[t+1], saida em C[t+K]. Dividendo de quem ja estava posicionado
    entra como direito economico sem reinvestimento intrabloco. Ativos recebidos
    sao marcados diariamente e liquidados junto da origem, com custos/slippage.
    """
    for value in (slippage_bps, cost_bps):
        if not np.isfinite(value) or not 0 <= value < 10000:
            raise ValueError("Slippage/custos devem estar em [0,10000) pontos base por ponta.")
    origins = integers(origins, "Origens")
    if (k < 1 or not len(origins) or origins.min() < 1 or origins.max()+k > market.n
            or (len(origins) > 1 and np.any(np.diff(origins) != k))):
        raise ValueError("Blocos devem ser consecutivos, completos e separados por K pregoes.")
    if scores.shape != (len(origins), len(market.tickers)):
        raise ValueError("Scores/origens/universo incompativeis.")
    finite = np.isfinite(scores)
    if np.any(~finite.any(axis=1)):
        raise ValueError(f"Nenhuma previsao finita na origem {origins[~finite.any(axis=1)][0]}.")
    sign = 1 if strategy.lado == "compra" else -1
    eligible = finite & ((sign*scores > 0) if strategy.filtrar_sinal else True)
    # Universo alfabetico + sort estavel = empate deterministico sem usar precos futuros.
    ordered = np.argsort(np.where(eligible, -sign*scores, np.inf), axis=1, kind="stable")
    chosen = ordered[:, :min(strategy.n_ativos, scores.shape[1])]
    active = np.take_along_axis(eligible, chosen, axis=1)
    n = active.sum(axis=1)
    weights = active/np.maximum(n[:, None], 1)
    days = origins[:, None] + np.arange(k)
    a = market.arrays
    opening = a["abertura"][origins[:, None], chosen]
    open_ok = a["pode_executar_abertura"][origins[:, None], chosen]
    close_ok = a["pode_executar_fechamento"][(origins+k-1)[:, None], chosen]
    for label, ok, dates in [("ABERTURA_INEXECUTAVEL", open_ok, origins),
                             ("FECHAMENTO_INEXECUTAVEL", close_ok, origins+k-1)]:
        bad = active & ~ok
        if bad.any():
            i, j = np.argwhere(bad)[0]
            raise ValueError(f"{label}: {market.tickers[chosen[i,j]]}, "
                             f"date={dates[i]+1}, origem={origins[i]}. Sem substituir ativo.")
    row, col = days[:, :, None], chosen[:, None, :]
    c = np.where(active[:, None, :], a["fechamento"][row, col], 1.)
    factors = a["fator_quantidade_evento"][row, col].copy()
    factors[:, 0, :] = 1.  # Entrada na data ex NAO recebe o evento dessa data.
    scale = np.cumprod(factors, axis=1)
    previous_scale = np.concatenate([np.ones_like(scale[:, :1]), scale[:, :-1]], axis=1)
    div = a["provento_por_acao_com"][row, col]*previous_scale
    div[:, 0, :] = 0.
    assets = a["valor_ativos_por_acao_com"][row, col].copy()
    assets[:, 0, :] = 0.
    bad = (assets > 0) & active[:, None, :]
    for i, d, j in np.argwhere(bad):
        if (int(days[i, d]), int(chosen[i, j])) not in market.received:
            raise ValueError(f"ATIVO_DISTRIBUIDO_SEM_TRAJETORIA: {market.tickers[chosen[i,j]]}, "
                             f"date={days[i,d]+1}. Requer cotacoes/quantidades do ativo recebido.")
    div = np.cumsum(div, axis=1)
    opening = np.where(active, opening, 1.)
    slip, cost = slippage_bps/10000, cost_bps/10000
    entry_price = opening*(1+sign*slip)
    # Orçamento de entrada inclui a taxa: exposicao bruta <=1x para ambos os lados.
    quantity = weights/(entry_price*(1+cost))
    notional = quantity*entry_price
    entry_fee = cost*notional.sum(axis=1)
    cash = 1-sign*notional.sum(axis=1)-entry_fee
    exit_quote = a["fechamento_execucao"][(origins+k-1)[:, None], chosen]
    exit_price = np.where(active, exit_quote, 1.)*(1-sign*slip)
    mark = c.copy()
    mark[:, -1, :] = exit_price
    dividends = (quantity[:, None, :]*div).sum(axis=2)
    received_value = np.zeros((len(origins), k))
    received_cash = np.zeros_like(received_value)
    received_fee = np.zeros(len(origins))
    received_positions = []
    # Poucos eventos: laço apenas sobre posicoes que atravessam a data ex.
    for i, d, j in np.argwhere(bad):
        for receipt in market.received[(int(days[i, d]), int(chosen[i, j]))]:
            length = k-int(d)
            if length > len(receipt["close"]):
                raise ValueError(f"TRAJETORIA_ATIVO_RECEBIDO_INCOMPLETA: {receipt['ticker']}, "
                                 f"saida date={origins[i]+k}, evento={receipt['id']}.")
            if not receipt["exit_ok"][length-1]:
                raise ValueError(f"FECHAMENTO_ATIVO_RECEBIDO_INEXECUTAVEL: {receipt['ticker']}, "
                                 f"date={origins[i]+k}, evento={receipt['id']}.")
            q_received = quantity[i, j]*previous_scale[i, d, j]*receipt["ratio"]
            marks = receipt["close"][:length].copy()
            exit_received = receipt["exit"][length-1]*(1-sign*slip)
            marks[-1] = exit_received
            q_exit = q_received*receipt["scale"][length-1]
            received_value[i, d:] += q_received*marks*receipt["scale"][:length]
            received_cash[i, d:] += q_received*receipt["cash"][:length]
            received_fee[i] += cost*q_exit*exit_received
            received_positions.append({"origin_step": origins[i], "entry_step": days[i, d]+1,
                "target_step": origins[i]+k, "ticker": receipt["ticker"], "rank": np.nan,
                "score_log_acumulado": np.nan, "peso_entrada": 0.,
                "quantidade_entrada_por_capital": sign*q_received,
                "quantidade_saida_por_capital": sign*q_exit, "preco_entrada": 0.,
                "preco_saida": exit_received, "direitos_por_capital": sign*q_received*receipt["cash"][length-1],
                "tipo_posicao": "RECEBIDA", "ticker_origem": market.tickers[chosen[i, j]],
                "id_distribuicao": receipt["id"]})
    dividends += received_cash
    equity = cash[:, None]+sign*((quantity[:, None, :]*mark*scale).sum(axis=2)+received_value+dividends)
    exit_fee = cost*(quantity*scale[:, -1, :]*exit_price).sum(axis=1)
    exit_fee += received_fee
    equity[:, -1] -= exit_fee
    if not np.isfinite(equity).all() or np.any(equity <= 0):
        raise ValueError("PATRIMONIO_NAO_POSITIVO: carteira insolvente; nao truncar perdas nem continuar alavancando.")
    previous = np.concatenate([np.ones((len(origins), 1)), equity[:, :-1]], axis=1)
    returns = (equity/previous-1).ravel()
    wealth = np.cumprod(1+returns)
    if not np.isfinite(wealth).all() or np.any(wealth <= 0):
        raise ValueError("Patrimonio acumulado fora do intervalo numerico.")
    data = market.calendar.date_pregao.values
    daily = pd.DataFrame({"date": days.ravel()+1, "data": data[days.ravel()],
        "origin_step": np.repeat(origins, k), "n_ativos": np.repeat(n, k),
        "retorno": returns, "rf": market.rf[days.ravel()], "patrimonio": wealth,
        "drawdown": wealth/np.maximum.accumulate(np.r_[1., wealth])[1:]-1,
        "direitos_acumulados_por_capital_bloco": sign*dividends.ravel(),
        "valor_ativos_recebidos_por_capital_bloco": sign*received_value.ravel()})
    periods = pd.DataFrame({"origin_step": origins, "entry_step": origins+1, "target_step": origins+k,
        "data_sinal": data[origins-1], "data_entrada": data[origins], "data_saida": data[origins+k-1],
        "n_ativos": n, "retorno": equity[:, -1]-1, "patrimonio": wealth.reshape(-1, k)[:, -1],
        "custo_entrada_por_capital": entry_fee, "custo_saida_por_capital": exit_fee,
        "direitos_por_capital": sign*dividends[:, -1],
        "valor_saida_ativos_recebidos_por_capital": sign*received_value[:, -1],
        "custo_saida_ativos_recebidos_por_capital": received_fee,
        "retorno_cdi": np.expm1(np.log1p(market.rf[days]).sum(axis=1))})
    b, j = np.where(active)
    positions = pd.DataFrame({"origin_step": origins[b], "entry_step": origins[b]+1,
        "target_step": origins[b]+k, "ticker": market.tickers[chosen[b, j]],
        "rank": j+1, "score_log_acumulado": scores[b, chosen[b, j]], "peso_entrada": weights[b, j],
        "quantidade_entrada_por_capital": sign*quantity[b, j],
        "quantidade_saida_por_capital": sign*quantity[b, j]*scale[b, -1, j],
        "preco_entrada": entry_price[b, j], "preco_saida": exit_price[b, j],
        "direitos_por_capital": sign*quantity[b, j]*div[b, -1, j],
        "tipo_posicao": "SINAL", "ticker_origem": market.tickers[chosen[b, j]],
        "id_distribuicao": ""})
    if received_positions:
        positions = pd.concat([positions, pd.DataFrame(received_positions)], ignore_index=True)
    result = {"carteira_diaria": daily, "carteira_periodos": periods,
              "posicoes": positions, "metricas": metrics(daily, periods)}
    for frame in result.values():
        frame["k"] = k
        frame["estrategia"] = strategy.id
        frame["lado"] = strategy.lado
        frame["n_ativos_max"] = strategy.n_ativos
        frame["filtrar_sinal"] = strategy.filtrar_sinal
    return result
