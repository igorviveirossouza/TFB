"""MASTER data contracts, economic target alignment and temporal leakage checks.

The small fixture follows the R export schema; it is not the empirical B3 panel.
No model training or remote cluster access is required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_benchmark.data.master import load_master_bundle
from ts_benchmark.data.master_dataset import (
    MasterStandardizer,
    MasterWindowDataset,
    prepare_master_datasets,
    temporal_split,
)


class ExportFixture:
    """A complete, deliberately unordered, R-shaped export with known values."""

    def __init__(self, root: Path, fmt="csv"):
        self.root = root
        self.directory = root / "features_master_v1"
        self.directory.mkdir(parents=True)
        self.returns = root / "b3_daily_tfb_new.csv"
        self.fmt = fmt
        self.n = 48
        self.assets = ("ZZZZ3", "AAAA4", "TEST11")
        self.dates = pd.bdate_range("2024-01-03", periods=self.n)
        self.names = {
            "ativos": f"ativos_features.{fmt}",
            "mercado": f"mercado_features.{fmt}",
            "insumos": f"insumos_ativos_auditados.{fmt}",
        }
        t = np.arange(self.n)
        self.target = np.column_stack([
            .01 * np.sin(t / 4) + i * .001 for i in range(3)
        ])
        targets, rows, insumos = [], [], []
        for i, asset in enumerate(self.assets):
            rolling = pd.Series(self.target[:, i]).rolling(3).mean()
            for j, date in enumerate(self.dates):
                targets.append({"date": j + 1, "data": self.target[j, i], "cols": asset})
                candle = np.nan if (i == 1 and j == 11) else .005 * np.cos(j / 7) + i * .001
                activity = np.nan if j < 3 else 1 + .2 * np.sin(j) + i * .1
                if i == 0 and j == 8:
                    activity = 0.0
                rows.append({
                    "date": j + 1, "date_pregao": date.strftime("%Y-%m-%d"), "cols": asset,
                    "retorno_log": self.target[j, i], "retorno_media_3": rolling.iloc[j],
                    "flag_aquecimento_retornos_3": j < 2,
                    "ohlcv_log_close_open": candle, "ohlcv_posicao_close": .5,
                    "flag_ohlcv_disponivel": bool(np.isfinite(candle)),
                    "atividade_financeiro_relativo_3": activity,
                    "flag_financeiro_disponivel": bool(np.isfinite(activity)),
                })
                insumos.append({
                    "date": j + 1, "date_pregao": date.strftime("%Y-%m-%d"), "cols": asset,
                    "fator_quantidade_acumulado": 1.0, "volume_financeiro": 1e5 + j * 100,
                    "quantidade_ajustada": 1000 + j, "numero_negocios": 20 + j,
                    "cotacao_observada": bool(np.isfinite(candle)),
                })
        # Source first-occurrence order is deliberately different from R universo order.
        pd.DataFrame(targets).to_csv(self.returns, index=False)
        asset_frame = pd.DataFrame(rows).sample(frac=1, random_state=2021)
        self.write(self.names["ativos"], asset_frame)
        market_mean = self.target.mean(axis=1)
        market = pd.DataFrame({
            "date": t + 1, "date_pregao": self.dates.strftime("%Y-%m-%d"),
            "mercado_media_log": market_mean,
            "mercado_momentum_3": pd.Series(market_mean).rolling(3).sum(),
            "flag_aquecimento_mercado_3": t < 2,
        }).iloc[::-1]
        self.write(self.names["mercado"], market)
        self.write(self.names["insumos"], pd.DataFrame(insumos).sample(frac=1, random_state=3))
        self.write("calendario_tfb.csv", pd.DataFrame({
            "date": t + 1, "date_pregao": self.dates.strftime("%Y-%m-%d"),
            "data_anterior": pd.DatetimeIndex([pd.Timestamp("2024-01-02"), *self.dates[:-1]]).strftime("%Y-%m-%d"),
        }))
        self.write("universo_ativos.csv", pd.DataFrame({"ordem": [1, 2, 3], "cols": sorted(self.assets)}))
        dictionary = []
        for block, frame in (("ativos", asset_frame), ("mercado", market)):
            for variable in frame.columns:
                if variable in {"date", "date_pregao", "cols"}:
                    kind, subblock, family = "chave", "identificacao", "chave"
                elif block == "mercado":
                    kind = "flag" if variable.startswith("flag_") else "feature"
                    subblock, family = "universo_congelado", "cross_section"
                else:
                    kind = "flag" if variable.startswith("flag_") else "feature"
                    subblock = ("retornos" if "retorno" in variable else
                                "ohlcv" if "ohlcv" in variable else "atividade")
                    family = "disponibilidade" if kind == "flag" else subblock
                dictionary.append({
                    "arquivo": self.names[block], "bloco": block, "subbloco": subblock,
                    "familia": family, "variavel": variable,
                    "janela": 3 if variable.endswith("_3") else np.nan,
                    "formula": "Fixture controlada; valores conhecidos",
                    "unidade": "booleano" if kind == "flag" else "adimensional", "tipo": kind,
                })
        self.write("dicionario_features.csv", pd.DataFrame(dictionary))
        self.refresh_quality()
        source_md5 = self.md5(self.returns)
        roles = ["RETORNOS_USADOS_NO_TREINO", "b3_daily_tfb.csv", "calendario_tfb.csv",
                 "auditoria_retornos_equity.csv", "auditoria_retornos_etf.csv", "status_exportacao.csv",
                 "checksums_saidas.csv", "exportar_features_master.R", "preparar_features_master.R",
                 "preparar_base_backtest.R"]
        self.write("fontes.csv", pd.DataFrame({
            "papel": roles,
            "arquivo": ["/workstation/not-mounted/" + role for role in roles],
            "md5": [source_md5, source_md5, *(["b" * 32] * (len(roles) - 2))],
        }))
        self.write("manifesto.csv", pd.DataFrame([{
            "versao": "2026-10-23-master-features-v1", "criado_em_utc": "2026-10-23 UTC",
            "arquivo_ativos": self.names["ativos"], "arquivo_mercado": self.names["mercado"],
            "n_linhas_ativos": self.n * 3, "n_ativos": 3, "n_pregoes": self.n,
            "inicio": self.dates[0].strftime("%Y-%m-%d"), "fim": self.dates[-1].strftime("%Y-%m-%d"),
            "janelas": "3", "retorno_modelo": "LOG_RETORNO_ECONOMICO_CONGELADO_SEM_RECALCULO",
            "mercado": "AGREGADOS_DO_PAINEL_FIXO_NAO_E_INDICE_EXTERNO",
            "quantidade": "NEGOCIADA_DIVIDIDA_POR_FATOR_QUANTIDADE_ACUMULADO_CAUSAL",
            "financeiro": "OBSERVADO_SEM_AJUSTE_POR_EVENTOS_OU_FATCOT",
            "ohlcv": "RAZOES_INTRADIARIAS_APENAS_COTACOES_OBSERVADAS_COHERENTES",
            "fracionario": "RETORNO_PRESERVADO_SEM_OHLC_ATIVIDADE_NESTA_VERSAO",
            "disponibilidade": "APOS_FECHAMENTO_T_PARA_PREVISOES_A_PARTIR_DE_T_MAIS_1",
            "normalizacao": "NAO_APLICADA_AJUSTAR_SOMENTE_NO_TREINO_EM_PYTHON",
            "ausencias": "NA_COM_MASCARA_SEM_BFILL_SEM_EXCLUIR_ATIVOS_OU_DATAS",
            "n_na_ativos": int(asset_frame.isna().sum().sum()),
            "n_na_mercado": int(market.isna().sum().sum()),
            "painel_md5": source_md5,
            # The original audit calendar need not be byte-identical after R rewrites it.
            "calendario_md5": "b" * 32, "situacao": "CONFERIDA_COM_AUSENCIAS_DOCUMENTADAS",
            "n_pregoes_historico": 0, "criterio_historico": "SEM_PREHISTORICO_COMPLETO",
        }]))
        self.rehash()

    @staticmethod
    def md5(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    def write(self, name, frame):
        path = self.directory / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            # R's write.csv emits TRUE/FALSE. Preserve that contract in fixtures.
            frame = frame.copy()
            for column in frame:
                if pd.api.types.is_bool_dtype(frame[column]):
                    frame[column] = frame[column].map({True: "TRUE", False: "FALSE"})
            frame.to_csv(path, index=False, na_rep="")

    def read(self, name):
        path = self.directory / name
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")

    def rehash(self):
        paths = sorted(p for p in self.directory.iterdir() if p.name != "checksums_saidas.csv")
        self.write("checksums_saidas.csv", pd.DataFrame({
            "arquivo": [p.name for p in paths], "md5": [self.md5(p) for p in paths],
        }))

    def refresh_quality(self):
        quality = []
        for block in ("ativos", "mercado"):
            frame = self.read(self.names[block])
            for variable in frame.columns.difference(["date", "date_pregao", "cols"], sort=False):
                values = frame[variable]
                quality.append({
                    "arquivo": self.names[block], "variavel": variable, "n_linhas": len(values),
                    "n_ausentes": int(values.isna().sum()), "n_finitos": int(values.notna().sum()),
                    "n_zeros": int((values == 0).sum()),
                })
        self.write("qualidade_features.csv", pd.DataFrame(quality))

    def load(self, **kwargs):
        return load_master_bundle(self.directory, self.returns, **kwargs)


@pytest.fixture
def exported(tmp_path):
    return ExportFixture(tmp_path)


def test_alignment_preserves_tfb_asset_order_and_real_calendar(exported):
    bundle = exported.load(expected_assets=3, expected_dates=48)
    assert tuple(bundle.assets) == exported.assets
    np.testing.assert_allclose(bundle.targets, exported.target, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(bundle.steps, np.arange(1, 49))
    pd.testing.assert_index_equal(pd.DatetimeIndex(bundle.dates), exported.dates, check_names=False)
    r_index = bundle.asset_features.index("retorno_log")
    np.testing.assert_allclose(bundle.asset_values[:, :, r_index], exported.target, atol=1e-15)
    assert bundle.asset_values.shape[:2] == (48, 3)
    assert bundle.asset_flags.dtype == bool
    flag = bundle.asset_flag_names.index("flag_aquecimento_retornos_3")
    assert bundle.asset_flags[:2, :, flag].all()
    assert not bundle.asset_flags[2:, :, flag].any()  # FALSE must not become bool("FALSE").
    assert not {"date", "date_pregao", "cols", "numero_negocios", "volume_financeiro"}.intersection(bundle.asset_features)
    # The source paths in fontes belong to another workstation and need not exist.
    json.dumps(bundle.report)


def test_requested_asset_order_reindexes_every_array(exported):
    default = exported.load()
    order = tuple(reversed(exported.assets))
    alternate = exported.load(asset_order=order)
    assert tuple(alternate.assets) == order
    np.testing.assert_array_equal(alternate.targets, default.targets[:, ::-1])
    np.testing.assert_array_equal(alternate.asset_values, default.asset_values[:, ::-1, :])
    np.testing.assert_array_equal(alternate.asset_flags, default.asset_flags[:, ::-1, :])
    for invalid in [("AAAA4",), ("AAAA4", "AAAA4", "TEST11"), ("AAAA4", "TEST11", "FORA3")]:
        with pytest.raises(ValueError):
            exported.load(asset_order=invalid)


def test_table_permutation_never_changes_values(exported):
    before = exported.load()
    for name in (exported.names["ativos"], exported.names["mercado"], "calendario_tfb.csv", "universo_ativos.csv"):
        exported.write(name, exported.read(name).sample(frac=1, random_state=8))
    exported.rehash()
    after = exported.load()
    np.testing.assert_array_equal(before.targets, after.targets)
    np.testing.assert_array_equal(before.asset_values, after.asset_values)
    np.testing.assert_array_equal(before.market_values, after.market_values)


@pytest.mark.parametrize("kwargs", [{"expected_assets": 55}, {"expected_dates": 3664},
                                    {"asset_blocks": ("unknown",)}])
def test_wrong_experiment_contract_is_rejected(exported, kwargs):
    with pytest.raises(ValueError):
        exported.load(**kwargs)


def test_hash_tampering_fails_even_if_only_disabled_market_changes(exported):
    path = exported.directory / exported.names["mercado"]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        exported.load(asset_blocks=("retornos",), use_market=False)


def test_required_file_cannot_be_omitted_from_checksum_manifest(exported):
    hashes = exported.read("checksums_saidas.csv")
    hashes = hashes[hashes.arquivo != "dicionario_features.csv"]
    exported.write("checksums_saidas.csv", hashes)
    with pytest.raises(ValueError):
        exported.load()


def test_external_paths_in_checksums_are_rejected(exported):
    hashes = exported.read("checksums_saidas.csv")
    hashes.loc[0, "arquivo"] = "../b3_daily_tfb_new.csv"
    hashes.loc[0, "md5"] = exported.md5(exported.returns)
    exported.write("checksums_saidas.csv", hashes)
    with pytest.raises(ValueError):
        exported.load()


def test_source_target_file_identity_is_checked(exported):
    exported.returns.write_text(exported.returns.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        exported.load()


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra", "return", "date", "inf", "unknown_column"])
def test_rehashed_invalid_asset_table_is_rejected(exported, fault):
    name = exported.names["ativos"]
    frame = exported.read(name)
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "missing":
        frame = frame.iloc[1:]
    elif fault == "extra":
        frame.loc[0, "cols"] = "FORA3"
    elif fault == "return":
        frame.loc[0, "retorno_log"] += .01
    elif fault == "date":
        frame.loc[0, "date_pregao"] = "2024-01-01"
    elif fault == "inf":
        frame.loc[0, "ohlcv_log_close_open"] = np.inf
    elif fault == "unknown_column":
        frame["unregistered_price"] = 5000.0
    exported.write(name, frame)
    exported.refresh_quality()
    exported.rehash()
    with pytest.raises(ValueError):
        exported.load()


@pytest.mark.parametrize("fault", ["duplicate", "missing", "wrong_type", "wrong_block"])
def test_dictionary_contract_cannot_be_bypassed(exported, fault):
    dictionary = exported.read("dicionario_features.csv")
    if fault == "duplicate":
        dictionary = pd.concat([dictionary, dictionary.iloc[[0]]], ignore_index=True)
    elif fault == "missing":
        dictionary = dictionary[dictionary.variavel != "ohlcv_log_close_open"]
    elif fault == "wrong_type":
        dictionary.loc[dictionary.variavel == "date", "tipo"] = "feature"
    elif fault == "wrong_block":
        dictionary.loc[dictionary.variavel == "ohlcv_log_close_open", "bloco"] = "mercado"
    exported.write("dicionario_features.csv", dictionary)
    exported.rehash()
    with pytest.raises(ValueError):
        exported.load()


def test_duplicate_csv_header_is_not_silently_mangled(exported):
    path = exported.directory / exported.names["ativos"]
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    header[-1] = header[-2]
    lines[0] = ",".join(header)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    exported.rehash()
    with pytest.raises(ValueError):
        exported.load()


def test_invalid_boolean_does_not_become_true(exported):
    frame = exported.read(exported.names["ativos"])
    frame["flag_ohlcv_disponivel"] = frame["flag_ohlcv_disponivel"].astype(object)
    frame.loc[0, "flag_ohlcv_disponivel"] = "unknown"
    exported.write(exported.names["ativos"], frame)
    exported.rehash()
    with pytest.raises(ValueError):
        exported.load()


def test_calendar_chain_is_checked_after_hashes(exported):
    calendar = exported.read("calendario_tfb.csv")
    calendar.loc[10, "data_anterior"] = calendar.loc[8, "date_pregao"]
    exported.write("calendario_tfb.csv", calendar)
    exported.rehash()
    with pytest.raises(ValueError):
        exported.load()


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="Optional Parquet engine unavailable")
def test_csv_and_parquet_exports_have_identical_contracts(tmp_path):
    csv_export = ExportFixture(tmp_path / "csv")
    parquet_export = ExportFixture(tmp_path / "parquet", "parquet")
    csv_bundle, parquet_bundle = csv_export.load(), parquet_export.load()
    for field in ("targets", "asset_values", "market_values", "asset_flags", "market_flags", "steps"):
        np.testing.assert_allclose(getattr(csv_bundle, field), getattr(parquet_bundle, field), rtol=0, atol=1e-15, equal_nan=True)
    assert csv_bundle.asset_features == parquet_bundle.asset_features
    assert csv_bundle.asset_flag_names == parquet_bundle.asset_flag_names


def test_standardizer_uses_observed_internal_train_only_and_neutral_missing(exported):
    bundle = exported.load()
    original = bundle.asset_values.copy()
    scaler = MasterStandardizer().fit(bundle, train_end=30)
    prepared = scaler.transform(bundle)
    expected_mean = np.nanmean(bundle.asset_values[:30], axis=0)
    expected_scale = np.nanstd(bundle.asset_values[:30], axis=0, ddof=0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(scaler.asset_mean, expected_mean)
    np.testing.assert_allclose(scaler.asset_scale, expected_scale)
    np.testing.assert_array_equal(scaler.asset_count, np.isfinite(bundle.asset_values[:30]).sum(axis=0))
    np.testing.assert_array_equal(prepared.asset_observed, np.isfinite(original))
    assert np.isfinite(prepared.asset_values).all()
    assert (prepared.asset_values[~prepared.asset_observed] == 0).all()
    constant = bundle.asset_features.index("ohlcv_posicao_close")
    np.testing.assert_array_equal(prepared.asset_values[:, :, constant], np.zeros((48, 3)))
    np.testing.assert_array_equal(bundle.asset_values, original)  # No input mutation.
    np.testing.assert_array_equal(prepared.bundle.targets, exported.target)
    state = scaler.state_dict()
    assert state["train_end"] == 30
    json.dumps(state, allow_nan=False)


def test_future_extremes_do_not_change_train_scaler_or_training_windows(exported):
    bundle = exported.load()
    ordinary = prepare_master_datasets(bundle, lookback=5, horizon=3)
    fit = ordinary.split.n_train
    assets = bundle.asset_values.copy()
    market = bundle.market_values.copy()
    targets = bundle.targets.copy()
    assets[fit:] = np.where(np.isfinite(assets[fit:]), 1e8, np.nan)
    market[fit:] = np.where(np.isfinite(market[fit:]), -1e9, np.nan)
    targets[fit:] += 1e6
    changed = prepare_master_datasets(replace(bundle, asset_values=assets, market_values=market, targets=targets),
                                      lookback=5, horizon=3)
    # Different array memory layouts may change floating-point reductions at the
    # last bit. Future information must not change their numerical values.
    before, after = ordinary.scaler.state_dict(), changed.scaler.state_dict()
    for field in ("train_end", "schema", "ddof", "missing_policy"):
        assert before[field] == after[field]
    for field in ("asset_mean", "asset_scale", "asset_count", "market_mean", "market_scale", "market_count"):
        np.testing.assert_allclose(before[field], after[field], rtol=1e-14, atol=1e-15)
    assert len(ordinary.train) == len(changed.train)
    for i in range(len(ordinary.train)):
        for field in ("asset_values", "market_values", "target", "asset_observed", "market_observed"):
            np.testing.assert_allclose(ordinary.train[i][field], changed.train[i][field], rtol=1e-14, atol=1e-15)


def test_zero_observation_and_unknown_have_different_masks(exported):
    bundle = exported.load()
    feature = bundle.asset_features.index("atividade_financeiro_relativo_3")
    prepared = MasterStandardizer().fit(bundle, 30).transform(bundle)
    assert bundle.asset_values[8, 0, feature] == 0
    assert prepared.asset_observed[8, 0, feature]
    assert not prepared.asset_observed[0, 0, feature]
    assert prepared.asset_values[0, 0, feature] == 0


@pytest.mark.parametrize("use_market", [True, False])
def test_nonbinary_constant_features_keep_unit_scale_with_missing_values(exported, use_market):
    """A rounded mean must not turn a constant into an almost-zero scale."""
    bundle = exported.load(use_market=use_market)
    feature = bundle.asset_features.index("ohlcv_log_close_open")
    assets = bundle.asset_values.copy()
    assets[:, :, feature] = .1
    for asset, missing_count in enumerate((1, 3, 7)):
        assets[:missing_count, asset, feature] = np.nan
    assets[33, :, feature] += 1e-6
    market = bundle.market_values.copy()
    if use_market:
        market_feature = bundle.market_features.index("mercado_momentum_3")
        market[:, market_feature] = .3
        market[:2, market_feature] = np.nan
        market[33, market_feature] += 1e-6
    constant_bundle = replace(bundle, asset_values=assets, market_values=market)
    scaler = MasterStandardizer().fit(constant_bundle, 33)
    prepared = scaler.transform(constant_bundle)
    np.testing.assert_array_equal(scaler.asset_mean[:, feature], np.full(3, .1))
    np.testing.assert_array_equal(scaler.asset_scale[:, feature], np.ones(3))
    np.testing.assert_array_equal(scaler.asset_count[:, feature], [32, 30, 26])
    np.testing.assert_array_equal(prepared.asset_values[:33, :, feature], np.zeros((33, 3)))
    np.testing.assert_array_equal(prepared.asset_observed, np.isfinite(assets))
    np.testing.assert_allclose(prepared.asset_values[33, :, feature], 1e-6, rtol=0, atol=1e-16)
    if use_market:
        assert scaler.market_mean[market_feature] == .3
        assert scaler.market_scale[market_feature] == 1
        assert scaler.market_count[market_feature] == 31
        np.testing.assert_array_equal(prepared.market_values[:33, market_feature], np.zeros(33))
        np.testing.assert_allclose(prepared.market_values[33, market_feature], 1e-6, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(prepared.market_observed, np.isfinite(market))
    else:
        assert prepared.market_values.shape == prepared.market_observed.shape == (48, 0)
    np.testing.assert_array_equal(prepared.bundle.targets, bundle.targets)


@pytest.mark.parametrize("which", ["asset", "market"])
def test_all_missing_train_dimension_never_borrows_validation_values(exported, which):
    bundle = exported.load()
    if which == "asset":
        values = bundle.asset_values.copy()
        values[:30, 1, 0] = np.nan
        bundle = replace(bundle, asset_values=values)
    else:
        values = bundle.market_values.copy()
        values[:30, 0] = np.nan
        bundle = replace(bundle, market_values=values)
    with pytest.raises(ValueError):
        MasterStandardizer().fit(bundle, 30)


def test_transform_rejects_different_asset_order(exported):
    scaler = MasterStandardizer().fit(exported.load(), 30)
    with pytest.raises(ValueError):
        scaler.transform(exported.load(asset_order=tuple(reversed(exported.assets))))


def test_multistep_windows_respect_each_label_boundary(exported):
    bundle = exported.load()
    datasets = prepare_master_datasets(bundle, lookback=5, horizon=3)
    assert datasets.split.n_pretest == 38
    assert datasets.split.n_train == 33
    expected_starts = {"train": np.arange(5, 31), "validation": np.arange(33, 36), "test": np.arange(38, 46)}
    for split_name, starts in expected_starts.items():
        dataset = getattr(datasets, split_name)
        np.testing.assert_array_equal(dataset.target_starts, starts)
        assert len(dataset) == len(starts)
        for k in (0, len(dataset) - 1):
            row, j = dataset[k], starts[k]
            assert row["asset_values"].shape == (5, 3, len(bundle.asset_features))
            assert row["market_values"].shape == (5, len(bundle.market_features))
            assert row["target"].shape == (3, 3)
            np.testing.assert_array_equal(row["target"], bundle.targets[j:j + 3])
            np.testing.assert_array_equal(row["asset_values"], datasets.prepared.asset_values[j - 5:j])
            np.testing.assert_array_equal(row["asset_flags"], bundle.asset_flags[j - 5:j])
            assert row["origin_step"] == j
            np.testing.assert_array_equal(row["target_steps"], np.arange(j + 1, j + 4))
            assert row["origin_date"] == exported.dates[j - 1].strftime("%Y-%m-%d")


def test_ablations_preserve_every_training_validation_test_origin(exported):
    full = prepare_master_datasets(exported.load(), lookback=5, horizon=3)
    for blocks, market in [(("retornos",), False), (("retornos", "ohlcv"), True),
                           (("retornos", "atividade"), False)]:
        bundle = exported.load(asset_blocks=blocks, use_market=market)
        partial = prepare_master_datasets(bundle, lookback=5, horizon=3)
        assert bool(len(bundle.market_features)) is market
        if not market:
            assert bundle.market_values.shape == (48, 0)
            assert bundle.market_flags.shape == (48, 0)
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(getattr(full, name).target_starts, getattr(partial, name).target_starts)
            np.testing.assert_array_equal(getattr(full, name)[0]["target"], getattr(partial, name)[0]["target"])
        assert all("ohlcv" not in name for name in bundle.asset_features) if "ohlcv" not in blocks else True
        assert all("atividade" not in name for name in bundle.asset_features) if "atividade" not in blocks else True


@pytest.mark.parametrize("kwargs", [{"n_dates": 0}, {"n_dates": 48, "train_percentage": 100},
                                     {"n_dates": 48, "train_percentage": 80.5},
                                     {"n_dates": 48, "train_ratio_in_tv": 1}])
def test_invalid_temporal_splits_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        temporal_split(**kwargs)


def test_insufficient_validation_horizon_is_not_silently_shortened(exported):
    with pytest.raises(ValueError):
        prepare_master_datasets(exported.load(), lookback=5, horizon=6)


def test_cli_validates_prepares_windows_and_preserves_existing_report(exported):
    script = Path(__file__).resolve().parents[1] / "scripts/validate_master_data.py"
    report = exported.root / "validation.json"
    command = [sys.executable, str(script), "--bundle", str(exported.directory),
               "--returns", str(exported.returns), "--expected-assets", "3", "--expected-dates", "48",
               "--prepare-windows", "--lookback", "3", "--horizon", "2", "--report", str(report)]
    run = subprocess.run(command, cwd=exported.root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(report.read_text())
    assert result["status"] == "VALIDATED_AND_WINDOWED"
    assert result["n_assets"] == 3 and result["n_dates"] == 48
    assert result["split"]["n_train"] == 33 and result["split"]["n_pretest"] == 38
    assert result["windows"]["test"]["first_origin_step"] == 38
    assert result["windows"]["test"]["last_target_step"] == 48
    assert result["windows"]["test"]["count"] == 9
    assert result["windows"]["test"]["target_shape"] == [2, 3]
    report_before = report.read_bytes()
    repeated = subprocess.run(command, cwd=exported.root, capture_output=True, text=True)
    assert repeated.returncode == 1
    assert report.read_bytes() == report_before
    # A changed target source is rejected even in validation-only CLI mode.
    exported.returns.write_text(exported.returns.read_text() + "\n", encoding="utf-8")
    bad = subprocess.run(command[:command.index("--prepare-windows")],
                         cwd=exported.root, capture_output=True, text=True)
    assert bad.returncode == 1
    assert "FALHA" in bad.stderr
