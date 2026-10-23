"""Read the audited R MASTER bundle without filling, scaling or changing its panel.

The original ``date,data,cols`` return CSV is required, not TFB's subsequently
prepared synthetic-date CSV. All files are validated before selecting ablations.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "2026-10-23-master-features-v1"
ASSET_BLOCKS = ("retornos", "ohlcv", "atividade")
_MD5 = re.compile(r"[0-9a-fA-F]{32}\Z")


@dataclass
class MasterBundle:
    targets: np.ndarray
    asset_values: np.ndarray
    market_values: np.ndarray
    asset_features: tuple[str, ...]
    market_features: tuple[str, ...]
    asset_flags: np.ndarray
    market_flags: np.ndarray
    asset_flag_names: tuple[str, ...]
    market_flag_names: tuple[str, ...]
    steps: np.ndarray
    dates: pd.DatetimeIndex
    assets: tuple[str, ...]
    report: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def n_dates(self) -> int:
        return len(self.steps)

    @property
    def n_assets(self) -> int:
        return len(self.assets)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe bundle filename: {name!r}")
    path = root / name
    if path.resolve().parent != root or not path.is_file():
        raise ValueError(f"Missing bundle file or path outside bundle: {name}")
    return path


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or len(set(header)) != len(header) or any(not x.strip() for x in header):
            raise ValueError(f"Empty/duplicate CSV column names in {path.name}")
        frame = pd.read_csv(path, float_precision="round_trip", encoding="utf-8-sig")
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table format: {path.name}")
    if frame.empty or not frame.columns.is_unique:
        raise ValueError(f"Empty table or duplicate column names: {path.name}")
    return frame


def _columns(
    frame: pd.DataFrame, required: Sequence[str], name: str, *, exact=False
) -> None:
    missing = set(required) - set(frame.columns)
    if missing or (exact and set(frame.columns) != set(required)):
        raise ValueError(f"Invalid columns in {name}; required: {', '.join(required)}")


def _numbers(values: pd.Series, name: str, *, allow_na=False) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"Numeric values required for {name}, not booleans")
    try:
        result = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric values in {name}") from exc
    if np.isinf(result).any() or (not allow_na and np.isnan(result).any()):
        raise ValueError(f"Nonfinite/missing numeric values in {name}")
    return result


def _integers(values: pd.Series, name: str, *, minimum=1) -> np.ndarray:
    array = _numbers(values, name)
    if (array < minimum).any() or (array != np.floor(array)).any() or (array >= 2**63).any():
        raise ValueError(f"Invalid integer values in {name}")
    return array.astype(np.int64)


def _text(values: pd.Series, name: str) -> list[str]:
    if values.isna().any() or not values.map(
        lambda x: isinstance(x, str) and bool(x) and x == x.strip()
    ).all():
        raise ValueError(f"Missing/invalid text values in {name}")
    return values.tolist()


def _dates(values: pd.Series, name: str) -> pd.DatetimeIndex:
    # Arrow dates become Python date objects; CSV dates must be exact ISO dates.
    strings = values.astype(str)
    if not strings.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"Expected ISO YYYY-MM-DD dates in {name}")
    try:
        return pd.DatetimeIndex(pd.to_datetime(strings, format="%Y-%m-%d", errors="raise"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid calendar dates in {name}") from exc


def _booleans(values: pd.Series, name: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"Missing flag values in {name}")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    if not values.map(lambda x: isinstance(x, str) and x.upper() in {"TRUE", "FALSE"}).all():
        raise ValueError(f"Expected TRUE/FALSE flags in {name}")
    return values.str.upper().eq("TRUE").to_numpy()


def _count(value: Any, name: str) -> int:
    return int(_integers(pd.Series([value]), name, minimum=0)[0])


def _align(
    frame: pd.DataFrame,
    steps: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: tuple[str, ...] | None,
    name: str,
) -> pd.DataFrame:
    keys = ["date"] if assets is None else ["date", "cols"]
    _columns(frame, keys, name)
    frame = frame.copy()
    frame["date"] = _integers(frame["date"], f"{name}/date")
    if assets is not None:
        _text(frame["cols"], f"{name}/cols")
        expected = pd.MultiIndex.from_product([steps, assets], names=keys)
    else:
        expected = pd.Index(steps, name="date")
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate keys in {name}")
    frame = frame.set_index(keys)
    if len(frame) != len(expected) or not frame.index.isin(expected).all():
        raise ValueError(f"Missing/extra dates or assets in {name}; panel must remain balanced")
    frame = frame.reindex(expected)
    if "date_pregao" in frame:
        actual_dates = _dates(frame["date_pregao"], f"{name}/date_pregao")
        expected_dates = dates if assets is None else dates.repeat(len(assets))
        if not np.array_equal(actual_dates.to_numpy(), expected_dates.to_numpy()):
            raise ValueError(f"Trading dates disagree with calendar in {name}")
    return frame.reset_index()


def _validate_dictionary(
    dictionary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    files: dict[str, str],
) -> None:
    required = [
        "arquivo", "bloco", "subbloco", "familia", "variavel", "janela",
        "formula", "unidade", "tipo",
    ]
    _columns(dictionary, required, "dictionary")
    for name in set(required) - {"janela"}:
        _text(dictionary[name], f"dictionary/{name}")
    if (
        dictionary.duplicated(["arquivo", "variavel"]).any()
        or set(dictionary["arquivo"]) != set(files.values())
    ):
        raise ValueError("Duplicate or unknown file/variable entries in dictionary")
    if not dictionary["tipo"].isin(["chave", "feature", "flag"]).all():
        raise ValueError("Unknown dictionary column type")
    windows = dictionary["janela"].dropna()
    if len(windows):
        _integers(windows, "dictionary/janela", minimum=2)
    for block, table in tables.items():
        rows = dictionary[dictionary["arquivo"] == files[block]]
        keys = {"date", "date_pregao", "cols"} if block == "ativos" else {"date", "date_pregao"}
        if set(rows["variavel"]) != set(table.columns) or not rows["bloco"].eq(block).all():
            raise ValueError(f"Dictionary does not describe every column of {block}")
        if set(rows.loc[rows["tipo"] == "chave", "variavel"]) != keys:
            raise ValueError(f"Identifiers must be dictionary keys, not model features: {block}")
        nonkeys = rows[rows["tipo"] != "chave"]
        allowed = ASSET_BLOCKS if block == "ativos" else ("universo_congelado",)
        if not nonkeys["subbloco"].isin(allowed).all():
            raise ValueError(f"Unknown feature subblock in {block}")
        raw_inputs = {
            "open", "high", "low", "close", "volume", "preco_unitario",
            "traded_contracts", "trade_quantity", "numero_negocios",
            "volume_financeiro", "quantidade_ajustada", "fator_quantidade_acumulado",
        }
        if raw_inputs.intersection(nonkeys["variavel"]):
            raise ValueError("Raw audited inputs/transaction counts are not v1 model features")
        if not rows.loc[rows["tipo"] == "chave", "subbloco"].eq("identificacao").all():
            raise ValueError(f"Invalid key subblock in {block}")
        for row in nonkeys.itertuples(index=False):
            column = table[row.variavel]
            if row.tipo == "flag":
                table[row.variavel] = _booleans(column, f"{block}/{row.variavel}")
            else:
                table[row.variavel] = _numbers(column, f"{block}/{row.variavel}", allow_na=True)
    ret = dictionary[(dictionary["bloco"] == "ativos") & (dictionary["variavel"] == "retorno_log")]
    if len(ret) != 1 or ret.iloc[0]["tipo"] != "feature" or ret.iloc[0]["subbloco"] != "retornos":
        raise ValueError("retorno_log must be a numeric return feature")


def load_master_bundle(
    bundle_dir: str | Path,
    returns_path: str | Path,
    *,
    asset_blocks: Sequence[str] = ASSET_BLOCKS,
    use_market: bool = True,
    asset_order: Sequence[str] | None = None,
    expected_assets: int | None = None,
    expected_dates: int | None = None,
    tolerance: float = 1e-12,
) -> MasterBundle:
    """Validate an immutable R export and select model inputs for an ablation.

    Asset order defaults to the first-occurrence order in the original return
    CSV, matching TFB's preparation. ``asset_order`` may explicitly reorder the
    entire same universe. Features retain NaN; flags remain separate booleans.
    """
    if (
        isinstance(asset_blocks, str)
        or not asset_blocks
        or len(set(asset_blocks)) != len(asset_blocks)
        or not set(asset_blocks).issubset(ASSET_BLOCKS)
    ):
        raise ValueError(f"asset_blocks must select distinct blocks from {ASSET_BLOCKS}")
    if not isinstance(use_market, bool) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid use_market/tolerance")
    root, returns_path = Path(bundle_dir).resolve(), Path(returns_path).resolve()
    checks_path = _file(root, "checksums_saidas.csv")
    checks_hash = _md5(checks_path)
    checks = _read(checks_path)
    _columns(checks, ["arquivo", "md5"], "checksums", exact=True)
    names = _text(checks["arquivo"], "checksums/arquivo")
    hashes = _text(checks["md5"], "checksums/md5")
    if (
        len(set(names)) != len(names)
        or "checksums_saidas.csv" in names
        or not all(_MD5.fullmatch(x) for x in hashes)
    ):
        raise ValueError("Invalid checksum manifest")
    paths = {name: _file(root, name) for name in names}
    snapshots = dict(zip(names, [x.lower() for x in hashes]))
    for name, path in paths.items():
        if _md5(path) != snapshots[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    required = {
        "manifesto.csv", "dicionario_features.csv", "qualidade_features.csv",
        "calendario_tfb.csv", "universo_ativos.csv", "fontes.csv",
    }
    if not required.issubset(paths):
        raise ValueError(f"Required files missing from checksums: {sorted(required - set(paths))}")
    manifest_frame = _read(paths["manifesto.csv"])
    fields = [
        "versao", "arquivo_ativos", "arquivo_mercado", "n_linhas_ativos",
        "n_ativos", "n_pregoes", "inicio", "fim", "painel_md5", "situacao",
        "n_na_ativos", "n_na_mercado",
    ]
    _columns(manifest_frame, fields, "manifest")
    if len(manifest_frame) != 1:
        raise ValueError("Expected one manifest row")
    manifest = manifest_frame.iloc[0].to_dict()
    if (
        manifest["versao"] != SCHEMA_VERSION
        or manifest["situacao"] not in {
            "CONFERIDA", "CONFERIDA_COM_AUSENCIAS_DOCUMENTADAS"
        }
    ):
        raise ValueError("Unsupported/unapproved MASTER export manifest")
    files = {"ativos": manifest["arquivo_ativos"], "mercado": manifest["arquivo_mercado"]}
    for block, name in files.items():
        expected_stem = "ativos_features" if block == "ativos" else "mercado_features"
        if (
            not isinstance(name, str)
            or Path(name).stem != expected_stem
            or Path(name).suffix not in {".csv", ".parquet"}
            or name not in paths
        ):
            raise ValueError(f"Invalid/untracked model table: {name}")
    suffix = Path(files["ativos"]).suffix
    insumos_name = "insumos_ativos_auditados" + suffix
    if Path(files["mercado"]).suffix != suffix or insumos_name not in paths:
        raise ValueError("Missing audited inputs or inconsistent model table formats")
    returns_hash = _md5(returns_path)
    if (
        not isinstance(manifest["painel_md5"], str)
        or returns_hash != manifest["painel_md5"].lower()
    ):
        raise ValueError(
            "Return CSV checksum differs from the frozen R input "
            "(use original b3_daily_tfb_new.csv)"
        )
    targets_frame = _read(returns_path)
    _columns(targets_frame, ["date", "data", "cols"], "return CSV", exact=True)
    original_order = tuple(dict.fromkeys(_text(targets_frame["cols"], "return CSV/cols")))
    universe = _read(paths["universo_ativos.csv"])
    _columns(universe, ["ordem", "cols"], "universe", exact=True)
    universe["ordem"] = _integers(universe["ordem"], "universe/ordem")
    universe = universe.sort_values("ordem")
    universe_assets = _text(universe["cols"], "universe/cols")
    if universe["cols"].duplicated().any() or not np.array_equal(
        universe["ordem"], np.arange(1, len(universe) + 1)
    ):
        raise ValueError("Invalid universe order or duplicate assets")
    if set(original_order) != set(universe_assets):
        raise ValueError("Return CSV and universe contain different assets")
    if isinstance(asset_order, str):
        raise ValueError("asset_order must be a sequence of full asset identifiers")
    assets = original_order if asset_order is None else tuple(asset_order)
    if (
        len(assets) != len(universe_assets)
        or len(set(assets)) != len(assets)
        or set(assets) != set(universe_assets)
    ):
        raise ValueError("asset_order must contain exactly the complete universe")
    calendar = _read(paths["calendario_tfb.csv"])
    _columns(calendar, ["date", "date_pregao", "data_anterior"], "calendar")
    calendar["date"] = _integers(calendar["date"], "calendar/date")
    calendar = calendar.sort_values("date")
    steps = np.arange(1, len(calendar) + 1, dtype=np.int64)
    if not np.array_equal(calendar["date"], steps):
        raise ValueError("Calendar indices must be consecutive from 1")
    dates = _dates(calendar["date_pregao"], "calendar/date_pregao")
    prior = _dates(calendar["data_anterior"], "calendar/data_anterior")
    if (
        not dates.is_monotonic_increasing
        or not dates.is_unique
        or (prior >= dates).any()
        or not np.array_equal(prior[1:].to_numpy(), dates[:-1].to_numpy())
    ):
        raise ValueError("Invalid trading calendar or previous-date linkage")
    t, n = len(steps), len(assets)
    for key, actual in [("n_ativos", n), ("n_pregoes", t), ("n_linhas_ativos", t*n)]:
        if _count(manifest[key], f"manifest/{key}") != actual:
            raise ValueError(f"Manifest dimension mismatch: {key}")
    for label, expected, actual in [("assets", expected_assets, n), ("dates", expected_dates, t)]:
        if expected is not None and (
            isinstance(expected, bool)
            or not isinstance(expected, (int, np.integer))
            or expected != actual
        ):
            raise ValueError(f"Expected {expected} {label}, found {actual}")
    bounds = _dates(pd.Series([manifest["inicio"], manifest["fim"]]), "manifest/date range")
    if bounds[0] != dates[0] or bounds[1] != dates[-1]:
        raise ValueError("Manifest dates do not match calendar")
    tables = {
        block: _align(
            _read(paths[name]), steps, dates,
            assets if block == "ativos" else None, name,
        )
        for block, name in files.items()
    }
    for block, table in tables.items():
        _columns(table, ["date_pregao"], files[block])
    targets_frame = _align(targets_frame, steps, dates, assets, "return CSV")
    targets = _numbers(targets_frame["data"], "return CSV/data").reshape(t, n)
    dictionary = _read(paths["dicionario_features.csv"])
    _validate_dictionary(dictionary, tables, files)
    returns_feature = tables["ativos"]["retorno_log"].to_numpy(dtype=float).reshape(t, n)
    limit = tolerance * np.maximum(1, np.maximum(np.abs(targets), np.abs(returns_feature)))
    if not np.isfinite(returns_feature).all() or not (
        np.abs(targets - returns_feature) <= limit
    ).all():
        raise ValueError("retorno_log differs from the frozen return targets")
    insumos = _align(_read(paths[insumos_name]), steps, dates, assets, insumos_name)
    _columns(
        insumos,
        ["date_pregao", "fator_quantidade_acumulado", "volume_financeiro",
         "quantidade_ajustada", "numero_negocios", "cotacao_observada"],
        insumos_name,
    )
    factor = _numbers(insumos["fator_quantidade_acumulado"], "quantity factor")
    if (factor <= 0).any():
        raise ValueError("Nonpositive accumulated quantity factor")
    for col in ["volume_financeiro", "quantidade_ajustada", "numero_negocios"]:
        if (_numbers(insumos[col], col, allow_na=True) < 0).any():
            raise ValueError(f"Negative audited activity: {col}")
    _booleans(insumos["cotacao_observada"], "audited quotation flag")
    quality = _read(paths["qualidade_features.csv"])
    _columns(
        quality,
        ["arquivo", "variavel", "n_linhas", "n_ausentes", "n_finitos", "n_zeros"],
        "quality",
    )
    expected_quality = {
        (row.arquivo, row.variavel)
        for row in dictionary.itertuples() if row.tipo != "chave"
    }
    actual_quality = list(zip(quality["arquivo"], quality["variavel"]))
    if len(set(actual_quality)) != len(actual_quality) or set(actual_quality) != expected_quality:
        raise ValueError("Quality report does not cover dictionary features and flags")
    file_tables = {files[block]: table for block, table in tables.items()}
    missing = {}
    for row in quality.itertuples(index=False):
        values = file_tables[row.arquivo][row.variavel]
        actual = [
            len(values), int(values.isna().sum()), int(values.notna().sum()),
            int((values == 0).sum()),
        ]
        recorded = [
            _count(getattr(row, key), f"quality/{key}")
            for key in ["n_linhas", "n_ausentes", "n_finitos", "n_zeros"]
        ]
        if actual != recorded:
            raise ValueError(f"Quality counts disagree for {row.arquivo}/{row.variavel}")
        missing[f"{row.arquivo}/{row.variavel}"] = actual[1]
    for block in tables:
        if _count(manifest[f"n_na_{block}"], f"manifest/n_na_{block}") != int(
            tables[block].isna().sum().sum()
        ):
            raise ValueError(f"Manifest missing-value count disagrees for {block}")
    has_missing = any(table.isna().any().any() for table in tables.values())
    if has_missing != (manifest["situacao"] == "CONFERIDA_COM_AUSENCIAS_DOCUMENTADAS"):
        raise ValueError("Manifest status disagrees with missing values")
    sources = _read(paths["fontes.csv"])
    _columns(sources, ["papel", "arquivo", "md5"], "sources")
    for col in ["papel", "arquivo", "md5"]:
        _text(sources[col], f"sources/{col}")
    recorded_input = sources[sources["papel"] == "RETORNOS_USADOS_NO_TREINO"]
    if len(recorded_input) != 1 or recorded_input.iloc[0]["md5"].lower() != returns_hash:
        raise ValueError("Source provenance differs from the frozen input")
    if not sources["md5"].map(lambda x: bool(_MD5.fullmatch(x))).all():
        raise ValueError("Invalid source hash")
    selections = {}
    for block in tables:
        rows = dictionary[(dictionary["bloco"] == block) & (dictionary["tipo"] != "chave")]
        if block == "ativos":
            rows = rows[rows["subbloco"].isin(asset_blocks)]
        elif not use_market:
            rows = rows.iloc[:0]
        for kind in ["feature", "flag"]:
            names_selected = tuple(rows.loc[rows["tipo"] == kind, "variavel"])
            shape = (t, n, len(names_selected)) if block == "ativos" else (t, len(names_selected))
            values = tables[block][list(names_selected)].to_numpy(
                dtype=bool if kind == "flag" else np.float64
            ).reshape(shape)
            selections[(block, kind)] = (names_selected, values)
    if not selections[("ativos", "feature")][0]:
        raise ValueError("Selected asset blocks contain no numeric features")
    if (
        _md5(checks_path) != checks_hash
        or _md5(returns_path) != returns_hash
        or any(_md5(paths[name]) != expected for name, expected in snapshots.items())
    ):
        raise RuntimeError("Input changed during validation; retry after its writer finishes")
    report = dict(
        schema_version=SCHEMA_VERSION, n_dates=t, n_assets=n,
        start_date=str(dates[0].date()), end_date=str(dates[-1].date()),
        asset_blocks=list(asset_blocks), use_market=use_market, assets=list(assets),
        asset_order_source="return_csv" if asset_order is None else "explicit",
        asset_features=list(selections[("ativos", "feature")][0]),
        market_features=list(selections[("mercado", "feature")][0]),
        missing_by_variable=missing, missing_policy="preserved_nan_no_rows_dropped",
        target_md5=returns_hash, checksums_md5=checks_hash,
    )
    provenance = dict(
        bundle_dir=str(root), returns_path=str(returns_path), manifest=manifest,
        files_md5=snapshots, sources=sources.to_dict(orient="records"),
    )
    return MasterBundle(
        targets=targets, asset_values=selections[("ativos", "feature")][1],
        market_values=selections[("mercado", "feature")][1],
        asset_features=selections[("ativos", "feature")][0],
        market_features=selections[("mercado", "feature")][0],
        asset_flags=selections[("ativos", "flag")][1],
        market_flags=selections[("mercado", "flag")][1],
        asset_flag_names=selections[("ativos", "flag")][0],
        market_flag_names=selections[("mercado", "flag")][0],
        steps=steps, dates=dates, assets=assets, report=report, provenance=provenance,
    )
