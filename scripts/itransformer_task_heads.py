"""Opt-in B3 worker support for matched iTransformer temporal/ranking heads.

Configuration and completion checks deliberately have no Torch dependency so
the same validation runs before submitting an array. The native B3 worker is
unchanged for configurations without task_head_mode.
"""
import hashlib
import json
import os
from pathlib import Path


RANKING_CONSTRUCTION = "restored_daily_projection_v1"
HEAD_KEYS = {"task_head_mode", "rank_windows", "ranking_construction"}


def settings(config, task=None):
    if task is not None and task["modelo"] != "iTransformer":
        return None
    defaults = config.get("model_defaults", {}).get("iTransformer", {})
    if not HEAD_KEYS.intersection(defaults):
        return None
    mode = defaults.get("task_head_mode")
    windows = defaults.get("rank_windows")
    if mode not in {"shared", "separate"}:
        raise ValueError("iTransformer task_head_mode must be shared or separate")
    if (not isinstance(windows, list) or not windows or
            any(type(k) is not int or k < 1 for k in windows) or len(set(windows)) != len(windows)):
        raise ValueError("iTransformer rank_windows must be unique positive integers")
    construction = defaults.get("ranking_construction", RANKING_CONSTRUCTION)
    if construction != RANKING_CONSTRUCTION:
        raise ValueError("Unsupported iTransformer ranking_construction")
    if any(k != 1 for k in config["trade_windows"]):
        raise ValueError("Task heads require K_LOSS_VALUES=1; RANK_WINDOWS_VALUES defines supervised windows")
    loss = config["loss"]
    if (config.get("source_data_kind", "log_return") != "log_return" or
            loss.get("loss_data_kind", "log_return") != "log_return" or
            not defaults.get("norm", True) or not loss.get("loss_inverse_norm", True)):
        raise ValueError("Restored iTransformer heads require normalized log-return inputs and inverse normalization")
    if loss.get("loss_score_kind", "log_return") not in {"log_return", "simple_return"}:
        raise ValueError("Unsupported restored score kind")
    if loss.get("momentum_mode", "off") != "off" or loss.get("momentum_input_mode", "off") != "off":
        raise ValueError("Momentum is an external benchmark in this task-head study")
    if any(not any(k <= h for k in windows) for h in config["horizons"]):
        raise ValueError("Each horizon needs at least one observable RANK_WINDOWS_VALUES window")
    result = dict(task_head_mode=mode, rank_windows=windows,
                  ranking_construction=construction)
    if task is not None:
        lam = task["cross_lambda"]
        if mode == "separate" and lam == 0:
            raise ValueError("Use the shared temporal control at lambda=0; a separate ranking head would be untrained")
        result["rank_windows"] = [k for k in windows if k <= task["pred_len"]]
        separate = mode == "separate" and lam > 0
        result.update(
            score_semantics="cumulative_" + (loss.get("loss_score_kind", "log_return")
                                              if separate else "log_return"),
            return_forecast_valid=not (mode == "separate" and lam == 1),
            ranking_scores_valid=separate,
            primary_file="ranking_scores.parquet" if separate else "part.parquet",
        )
    return result


def validate_config(config):
    contract = settings(config)
    if contract is not None:
        config["model_defaults"]["iTransformer"].update(contract)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_completed(output, contract, signature, primary_hash):
    """Never treat a missing/corrupt ranking file as a completed head fit."""
    path = Path(output) / "execution_metadata.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("design_sha256") != signature or any(record.get(k) != v for k, v in contract.items()):
        raise ValueError("Task-head completion metadata differs from the frozen design")
    hashes = record.get("files_sha256", {})
    required = {contract["primary_file"]}
    if contract["return_forecast_valid"]:
        required.add("part.parquet")
    if contract["ranking_scores_valid"]:
        required.add("ranking_scores.parquet")
    if not required.issubset(hashes) or hashes.get(contract["primary_file"]) != primary_hash:
        raise ValueError("Task-head completion is missing required artifact hashes")
    if any(not (Path(output) / name).is_file() or file_hash(Path(output) / name) != value
           for name, value in hashes.items()):
        raise ValueError("Task-head artifact is missing or differs from its completion hash")
    return True


def fit_and_export(config, task, hyper, panel, cut, output, *, step_offset=1):
    import numpy as np
    import pandas as pd
    import torch
    from b3_experiment_io import export_prediction_windows
    from ts_benchmark.baselines.itransformer_task_heads import ITransformerRestoredTaskAdapter
    from ts_benchmark.utils.random_utils import fix_all_random_seed

    contract = settings(config, task)
    if contract is None or step_offset != 1:
        raise ValueError("Task-head worker requires the explicit log-return contract")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fix_all_random_seed(config["strategy"]["seed"])
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    model = ITransformerRestoredTaskAdapter(**hyper)
    model.forecast_fit(panel.iloc[:cut], train_ratio_in_tv=config["strategy"]["train_ratio_in_tv"])
    h = task["pred_len"]
    origins = np.arange(cut, len(panel) - h + 1)
    if len(origins) < 1:
        raise ValueError("No held-out forecast origins")
    forecasts, rankings = [], []
    batch_size = int(hyper["batch_size"])

    class Maker:
        def __init__(self, selected):
            self.selected = selected

        def make_batch(self, batch_size, win_size):
            return dict(input=np.stack([panel.iloc[o-win_size:o].values for o in self.selected]),
                        time_stamps=np.stack([panel.index[o-win_size:o].values for o in self.selected]),
                        covariates=None)

    for start in range(0, len(origins), batch_size):
        selected = origins[start:start + batch_size]
        pred = model.batch_forecast(h, Maker(selected))
        shape = (len(selected), h, panel.shape[1])
        if pred.shape != shape or (contract["return_forecast_valid"] and not np.isfinite(pred).all()):
            raise ValueError("Invalid task-head forecast dimensions/values")
        if contract["return_forecast_valid"]:
            forecasts.append(pred)
        if contract["ranking_scores_valid"]:
            score = model.export_rank_scores
            if score.shape != shape or not np.isfinite(score).all():
                raise ValueError("Invalid restored ranking dimensions/values")
            rankings.append(score)
    hashes = {}
    if forecasts:
        actual = np.stack([panel.iloc[o:o+h].to_numpy() for o in origins])
        export_prediction_windows(actual, np.concatenate(forecasts), panel, cut, h, output,
                                  keep_csv=os.environ.get("MANTER_CSV", "FALSE").upper() == "TRUE",
                                  step_offset=step_offset)
        hashes["part.parquet"] = file_hash(output / "part.parquet")
    elif (output / "part.parquet").exists():
        raise ValueError("Separate lambda=1 destination contains an unqualified temporal forecast")
    if rankings:
        ks = contract["rank_windows"]
        rank = np.concatenate(rankings)[:, np.array(ks) - 1]
        frame = pd.DataFrame(rank.reshape(-1, panel.shape[1]), columns=panel.columns)
        frame.insert(0, "k_eval", np.tile(ks, len(origins)))
        frame.insert(0, "origin_step", np.repeat(origins, len(ks)))
        temporary = output / "ranking_scores.parquet.tmp"
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output / "ranking_scores.parquet")
        hashes["ranking_scores.parquet"] = file_hash(output / "ranking_scores.parquet")
    scaler = dict(mean=model.scaler.mean_.tolist(), scale=model.scaler.scale_.tolist())
    temporary = output / "task_head_checkpoint.pt.tmp"
    torch.save(dict(state_dict=model.check_point, scaler=scaler, hyper=hyper, task=task), temporary)
    os.replace(temporary, output / "task_head_checkpoint.pt")
    hashes["task_head_checkpoint.pt"] = file_hash(output / "task_head_checkpoint.pt")
    return dict(contract, files_sha256=hashes,
                parameter_count=sum(p.numel() for p in model.model.parameters()),
                scaler=scaler, torch_version=torch.__version__,
                device=str(next(model.model.parameters()).device))
