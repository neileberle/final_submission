"""Offline inference for the final TFT-TCN-Chronos ensemble."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models import Direct336TCN, get_tft_class


ID = "series_id"
TIME = "timestamp"
TARGET = "target"
KEYS = [ID, TIME]

TFT_COVARIATE_COLS = [
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
    "workload_intensity",
    "promotion_intensity",
    "nominal_capacity",
    "maintenance_known",
]
TFT_KNOWN_FUTURE = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
    "workload_intensity",
    "promotion_intensity",
    "shock_risk",
    "nominal_capacity",
]
TFT_STATIC_REALS = ["zone_sin", "zone_cos"]


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load checkpoints across PyTorch versions 2.2 through 2.5."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _make_asset_resolver(checkpoint_path: Path):
    """Resolve manifest-relative asset paths regardless of where the manifest sits.

    The checkpoint manifest stores paths like ``src/assets/tft/...`` relative to
    the submission root. Historically the manifest lived at that root, but it is
    also valid to keep it in a ``submission/`` subfolder (as the required command
    ``--checkpoint /submission/checkpoint.pt`` implies). Try a series of candidate
    base directories and use the first one that actually contains the asset.
    """
    checkpoint_path = checkpoint_path.resolve()
    package_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    for base in (
        checkpoint_path.parent,
        checkpoint_path.parent.parent,
        package_root,
        package_root.parent,
        Path.cwd(),
    ):
        base = base.resolve()
        if base not in candidates:
            candidates.append(base)

    def resolve(relative: str) -> Path:
        relative_path = Path(relative)
        for base in candidates:
            candidate = (base / relative_path).resolve()
            if candidate.exists():
                return candidate
        tried = "\n  ".join(str(base / relative_path) for base in candidates)
        raise FileNotFoundError(
            f"Could not locate manifest asset {relative!r}. Tried:\n  {tried}"
        )

    return resolve


def _release_accelerator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _mps_opt_in() -> bool:
    """Apple-silicon GPU is opt-in via ``DLAM_USE_MPS=1``.

    The graded run happens on a Linux CPU/CUDA container, where this never
    fires. Locally, MPS uses different kernels and reduction orders than CPU,
    so float32 results drift from the canonical CPU predictions recorded in
    LOCAL_VERIFICATION.json. Keep it off for any run whose numbers you intend
    to trust or report.
    """
    return os.environ.get("DLAM_USE_MPS", "0").strip().lower() in {"1", "true", "yes"}


def _device() -> torch.device:
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     if _mps_opt_in():
    #         return torch.device("mps")
    #     print(
    #         "Apple MPS is available but disabled (results would diverge from the "
    #         "canonical CPU predictions). Set DLAM_USE_MPS=1 to enable it.",
    #         flush=True,
    #     )
    return torch.device("cpu")


def _load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load observed history, known-future rows, and the exact forecast index."""
    input_dir = input_dir.resolve()
    index_path = next(
        (
            path
            for path in [
                input_dir / "forecast_index_test.csv",
                input_dir / "forecast_index_validation.csv",
            ]
            if path.exists()
        ),
        None,
    )
    if index_path is None:
        raise FileNotFoundError(
            "Expected forecast_index_test.csv or forecast_index_validation.csv"
        )
    forecast_index = pd.read_csv(index_path, parse_dates=[TIME])

    train_path = input_dir / "train.csv"
    validation_future_path = input_dir / "validation_input.csv"
    private_path = input_dir / "test_input.csv"

    if train_path.exists():
        history = pd.read_csv(train_path, parse_dates=[TIME])
        if validation_future_path.exists():
            future_source = pd.read_csv(validation_future_path, parse_dates=[TIME])
        elif private_path.exists():
            future_source = pd.read_csv(private_path, parse_dates=[TIME])
        else:
            raise FileNotFoundError(
                "train.csv was found, but no validation_input.csv or test_input.csv"
            )
    elif private_path.exists():
        private = pd.read_csv(private_path, parse_dates=[TIME])
        if TARGET not in private.columns:
            raise ValueError(
                "test_input.csv must contain observed target history. No target column found."
            )
        history = private.loc[private[TARGET].notna()].copy()
        future_source = private.drop(columns=[TARGET], errors="ignore")
    else:
        raise FileNotFoundError("Expected train.csv or test_input.csv in input_dir")

    if TARGET not in history.columns:
        raise ValueError("Observed history has no target column")
    history = history.loc[history[TARGET].notna()].sort_values(KEYS).reset_index(drop=True)
    future_source = future_source.sort_values(KEYS).drop_duplicates(KEYS, keep="last")
    future = forecast_index[KEYS].merge(
        future_source,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if forecast_index.duplicated(KEYS).any():
        raise ValueError("Forecast index contains duplicate keys")
    if history.duplicated(KEYS).any():
        raise ValueError("Observed history contains duplicate keys")
    counts = forecast_index.groupby(ID).size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 336:
        raise ValueError(f"Expected 336 forecast rows per series, got {counts.unique()}")


    # if set(forecast_index[ID].unique()) != set(history[ID].unique()):
    # This demands an exact set match between the series in your history file and the series in the forecast index. 
    # That's fine if train.csv always contains precisely the series being evaluated — but if the private test phase 
    # only asks you to forecast a subset of series (say, holds out some series entirely, or reuses the full course train.csv 
    # which may contain series beyond whatever this particular index covers), this line raises a ValueError on a perfectly valid input. You'd want:
    
    if not set(forecast_index[ID].unique()).issubset(history[ID].unique()):
        raise ValueError("History and forecast index contain different series IDs")
    return history, future, forecast_index


def _aligned_component(
    forecast_index: pd.DataFrame,
    frame: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    aligned = forecast_index[KEYS].merge(
        frame[KEYS + ["prediction"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(forecast_index):
        raise ValueError(f"{name}: wrong number of predictions")
    if aligned["prediction"].isna().any():
        raise ValueError(f"{name}: missing predictions after alignment")
    if not np.isfinite(aligned["prediction"]).all():
        raise ValueError(f"{name}: non-finite predictions")
    return aligned


def _prepare_tcn_arrays(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, int]]:
    means = pd.Series(metadata["means"], dtype=np.float64)
    stds = pd.Series(metadata["stds"], dtype=np.float64)
    medians = pd.Series(metadata["medians"], dtype=np.float64)
    hist_cols = list(metadata["hist_cols"])
    future_cols = list(metadata["future_cols"])
    series_ids = list(metadata["series_ids"])
    numeric_cols = list(means.index)
    base_future_cols = [column for column in numeric_cols if column != TARGET]
    missing_cols = [f"{column}__missing" for column in base_future_cols]

    missing_history = [column for column in numeric_cols if column not in history_raw]
    missing_future = [column for column in base_future_cols if column not in future_raw]
    if missing_history:
        raise ValueError(f"TCN history columns missing: {missing_history}")
    if missing_future:
        raise ValueError(f"TCN future columns missing: {missing_future}")

    history = history_raw.sort_values(KEYS).copy()
    for column, flag in zip(base_future_cols, missing_cols):
        history[flag] = history[column].isna().astype(np.float32)
    history[numeric_cols] = history.groupby(ID, sort=False)[numeric_cols].ffill()
    history[numeric_cols] = history[numeric_cols].fillna(medians[numeric_cols])
    history[numeric_cols] = (
        history[numeric_cols] - means[numeric_cols]
    ) / stds[numeric_cols]

    combined_covariates = pd.concat(
        [
            history_raw[KEYS + base_future_cols],
            future_raw[KEYS + base_future_cols],
        ],
        ignore_index=True,
    ).sort_values(KEYS)
    for column, flag in zip(base_future_cols, missing_cols):
        combined_covariates[flag] = combined_covariates[column].isna().astype(np.float32)
    combined_covariates[base_future_cols] = (
        combined_covariates.groupby(ID, sort=False)[base_future_cols]
        .ffill()
        .fillna(medians[base_future_cols])
    )
    combined_covariates[base_future_cols] = (
        combined_covariates[base_future_cols] - means[base_future_cols]
    ) / stds[base_future_cols]
    future = future_raw[KEYS].merge(
        combined_covariates[KEYS + future_cols],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    history_length = int(metadata["history"])
    horizon = int(metadata["horizon"])
    history_arrays: list[np.ndarray] = []
    future_arrays: list[np.ndarray] = []
    mapping = {sid: index for index, sid in enumerate(series_ids)}
    for sid in series_ids:
        hist = history.loc[history[ID].eq(sid)].sort_values(TIME)
        fut = future.loc[future[ID].eq(sid)].sort_values(TIME)
        if len(hist) < history_length:
            raise ValueError(
                f"TCN needs {history_length} history rows for {sid}; got {len(hist)}"
            )
        if len(fut) != horizon:
            raise ValueError(f"TCN needs {horizon} future rows for {sid}; got {len(fut)}")
        history_arrays.append(hist[hist_cols].to_numpy(np.float32)[-history_length:])
        future_arrays.append(fut[future_cols].to_numpy(np.float32))

    if np.isnan(np.stack(history_arrays)).any() or np.isnan(np.stack(future_arrays)).any():
        raise ValueError("TCN preprocessing produced NaN values")
    return history_arrays, future_arrays, mapping


def _predict_tcn(
    history: pd.DataFrame,
    future: pd.DataFrame,
    forecast_index: pd.DataFrame,
    checkpoint_paths: list[Path],
    device: torch.device,
) -> pd.DataFrame:
    print("[1/3] Predicting three-seed TCN ensemble ...", flush=True)
    first = _torch_load(checkpoint_paths[0], map_location="cpu")
    history_arrays, future_arrays, mapping = _prepare_tcn_arrays(history, future, first)
    series_ids = list(first["series_ids"])
    hist_cols = list(first["hist_cols"])
    future_cols = list(first["future_cols"])
    target_idx = hist_cols.index(TARGET)
    horizon = int(first["horizon"])
    means = pd.Series(first["means"])
    stds = pd.Series(first["stds"])
    del first

    x_hist = torch.from_numpy(np.stack(history_arrays))
    x_future = torch.from_numpy(np.stack(future_arrays))
    series_index = torch.tensor([mapping[sid] for sid in series_ids], dtype=torch.long)
    seed_predictions = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = _torch_load(checkpoint_path, map_location="cpu")
        if checkpoint["hist_cols"] != hist_cols or checkpoint["future_cols"] != future_cols:
            raise ValueError("TCN checkpoint schemas do not agree")
        model = Direct336TCN(
            hist_size=len(hist_cols),
            future_size=len(future_cols),
            n_series=len(series_ids),
            target_idx=target_idx,
            horizon=horizon,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(series_ids), 32):
                end = min(start + 32, len(series_ids))
                pred_norm = model(
                    x_hist[start:end].to(device),
                    x_future[start:end].to(device),
                    series_index[start:end].to(device),
                )
                pred_real = torch.clamp(
                    pred_norm * float(stds[TARGET]) + float(means[TARGET]),
                    min=0.0,
                )
                chunks.append(pred_real.cpu().numpy())
        seed_predictions.append(np.concatenate(chunks, axis=0))
        del model, checkpoint
        _release_accelerator()

    averaged = np.mean(seed_predictions, axis=0)
    rows = []
    for series_position, sid in enumerate(series_ids):
        required = forecast_index.loc[forecast_index[ID].eq(sid)].sort_values(TIME)
        for step, timestamp in enumerate(required[TIME]):
            rows.append(
                {ID: sid, TIME: timestamp, "prediction": float(averaged[series_position, step])}
            )
    return _aligned_component(forecast_index, pd.DataFrame(rows), "TCN")


def _impute_tft_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(KEYS).copy()
    missing = [column for column in TFT_COVARIATE_COLS if column not in result]
    if missing:
        raise ValueError(f"TFT covariate columns missing: {missing}")
    result[TFT_COVARIATE_COLS] = result.groupby(ID)[TFT_COVARIATE_COLS].transform(
        lambda values: values.ffill().bfill()
    )
    return result


def _predict_tft(
    history: pd.DataFrame,
    future: pd.DataFrame,
    forecast_index: pd.DataFrame,
    checkpoint_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    print("[2/3] Predicting TFT ...", flush=True)
    from pytorch_forecasting import TimeSeriesDataSet

    TFTFullOneCycle = get_tft_class()
    model = TFTFullOneCycle.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )
    model.eval()

    hist = _impute_tft_frame(history)
    fut = _impute_tft_frame(future)
    # Only use history values if a known-future column is entirely unavailable.
    last_known = hist.groupby(ID)[TFT_COVARIATE_COLS].last()
    for column in TFT_COVARIATE_COLS:
        if fut[column].isna().any():
            fut[column] = fut[column].fillna(fut[ID].map(last_known[column]))
    if fut[TFT_COVARIATE_COLS].isna().any().any():
        raise ValueError("TFT future-covariate imputation left NaN values")

    minimum_time = hist[TIME].min()
    hist["time_idx"] = (
        (hist[TIME] - minimum_time).dt.total_seconds() / 3600
    ).astype(int)
    fut["time_idx"] = (
        (fut[TIME] - minimum_time).dt.total_seconds() / 3600
    ).astype(int)
    fut[TARGET] = 0.0

    needed = [
        ID,
        TIME,
        "time_idx",
        TARGET,
        *sorted(set(TFT_KNOWN_FUTURE + TFT_STATIC_REALS)),
    ]
    missing_hist = [column for column in needed if column not in hist]
    missing_fut = [column for column in needed if column not in fut]
    if missing_hist or missing_fut:
        raise ValueError(
            f"TFT required columns missing; history={missing_hist}, future={missing_fut}"
        )
    combined = (
        pd.concat([hist[needed], fut[needed]], ignore_index=True)
        .drop_duplicates([ID, "time_idx"], keep="last")
        .sort_values([ID, "time_idx"])
        .reset_index(drop=True)
    )
    parameters = model.dataset_parameters
    prediction_dataset = TimeSeriesDataSet.from_parameters(
        parameters,
        combined,
        predict=True,
        stop_randomization=True,
    )
    prediction_loader = prediction_dataset.to_dataloader(
        train=False, batch_size=32, num_workers=0
    )
    accelerator = {"cuda": "gpu", "mps": "mps"}.get(device.type, "cpu")
    with torch.inference_mode():
        raw = model.predict(
            prediction_loader,
            mode="raw",
            return_index=True,
            trainer_kwargs={
                "accelerator": accelerator,
                "devices": 1,
                "precision": "32",
                "logger": False,
                "enable_progress_bar": False,
            },
        )
    prediction_tensor = (
        raw.output["prediction"]
        if isinstance(raw.output, dict)
        else raw.output.prediction
    )
    median = prediction_tensor.float().cpu().numpy()[..., 3]
    index_frame = raw.index.reset_index(drop=True)
    rows = []
    for row_number, row in index_frame.iterrows():
        sid = row[ID]
        start_idx = int(row["time_idx"])
        for step in range(median.shape[1]):
            rows.append(
                {
                    ID: sid,
                    "time_idx": start_idx + step,
                    "prediction": float(max(median[row_number, step], 0.0)),
                }
            )
    long = pd.DataFrame(rows)
    long[TIME] = minimum_time + pd.to_timedelta(long["time_idx"], unit="h")
    result = _aligned_component(forecast_index, long, "TFT")
    del model, raw, prediction_loader, prediction_dataset
    _release_accelerator()
    return result


def _find_quantile_column(frame: pd.DataFrame, quantile: float):
    for column in frame.columns:
        try:
            if abs(float(column) - quantile) < 1e-9:
                return column
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Quantile {quantile} missing from Chronos output")


def _load_packed_chronos_model(packed_dir: Path, device: torch.device):
    """Rebuild the LoRA-merged Chronos-2 model from its int8 packed archive.

    The submission ships ``chronos_int8/`` instead of base weights + LoRA
    adapter: the adapter was merged offline and the merged weights were
    quantised int8-symmetric per group of ``group_size`` elements
    (``quant_manifest.json``). Weights are restored to float32 here, so the
    model that comes out is numerically the merged fp32 model up to the
    quantisation error recorded in the manifest.
    """
    import json

    from safetensors.torch import load_file
    from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model

    packed_dir = Path(packed_dir)
    quant_manifest_path = packed_dir / "quant_manifest.json"
    weights_path = packed_dir / "model.safetensors"
    for required in (quant_manifest_path, weights_path, packed_dir / "config.json"):
        if not required.exists():
            raise FileNotFoundError(f"Missing packed Chronos asset: {required}")

    quant_manifest = json.loads(quant_manifest_path.read_text())
    if quant_manifest.get("format") != "chronos_int8_v1":
        raise ValueError(
            f"Unsupported Chronos quantisation format: {quant_manifest.get('format')!r}"
        )
    group_size = int(quant_manifest["group_size"])

    packed = load_file(str(weights_path))
    state: dict[str, torch.Tensor] = {}
    for name in quant_manifest["kept"]:
        if name not in packed:
            raise KeyError(f"Packed Chronos archive is missing kept tensor {name!r}")
        state[name] = packed[name].to(torch.float32)
    for name, meta in quant_manifest["quantized"].items():
        q = packed.get(f"{name}::q")
        scale = packed.get(f"{name}::s")
        if q is None or scale is None:
            raise KeyError(f"Packed Chronos archive is missing int8 payload for {name!r}")
        if q.shape[-1] != group_size:
            raise ValueError(
                f"{name!r}: group size {q.shape[-1]} does not match manifest {group_size}"
            )
        state[name] = (q.to(torch.float32) * scale.to(torch.float32)).reshape(
            tuple(meta["shape"])
        )
    del packed

    config = Chronos2CoreConfig.from_pretrained(str(packed_dir))
    model = Chronos2Model(config)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    model.eval()
    return model


def _predict_chronos(
    history: pd.DataFrame,
    future: pd.DataFrame,
    forecast_index: pd.DataFrame,
    packed_model_dir: Path,
    quantile: float,
    covariate_medians: dict[str, float],
    device: torch.device,
) -> pd.DataFrame:
    print("[3/3] Predicting fine-tuned Chronos-2 ...", flush=True)
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        from chronos.chronos2 import Chronos2Pipeline

    covariates = list(covariate_medians)
    missing_history = [column for column in covariates if column not in history]
    missing_future = [column for column in covariates if column not in future]
    if missing_history or missing_future:
        raise ValueError(
            f"Chronos covariates missing; history={missing_history}, future={missing_future}"
        )
    medians = pd.Series(covariate_medians, dtype=np.float64)
    combined_covariates = pd.concat(
        [history[KEYS + covariates], future[KEYS + covariates]],
        ignore_index=True,
    ).sort_values(KEYS)
    combined_covariates[covariates] = (
        combined_covariates.groupby(ID, sort=False)[covariates]
        .ffill()
        .fillna(medians)
    )
    chronos_history = history[KEYS + [TARGET]].merge(
        combined_covariates[KEYS + covariates],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    chronos_future = future[KEYS].merge(
        combined_covariates[KEYS + covariates],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if chronos_history.isna().any().any() or chronos_future.isna().any().any():
        raise ValueError("Chronos preprocessing left NaN values")

    # The LoRA adapter was merged into the base weights offline and the merged
    # model was shipped int8-quantised (submission archive size limit); it is
    # restored to float32 here. Merging offline also keeps inference
    # deterministic and wrapper-independent: the original overnight script
    # predicted before explicitly switching the just-trained model out of train
    # mode, so its Chronos cache contained dropout noise and is not a
    # reproducibility reference.
    merged_model = _load_packed_chronos_model(packed_model_dir, device)
    pipeline = Chronos2Pipeline(model=merged_model)
    raw = pipeline.predict_df(
        chronos_history,
        future_df=chronos_future,
        prediction_length=336,
        quantile_levels=[quantile],
        batch_size=8,
        id_column=ID,
        timestamp_column=TIME,
        target=TARGET,
    )
    quantile_column = _find_quantile_column(raw, quantile)
    frame = raw[KEYS + [quantile_column]].rename(
        columns={quantile_column: "prediction"}
    )
    frame["prediction"] = frame["prediction"].clip(lower=0.0)
    result = _aligned_component(forecast_index, frame, "Chronos")
    del pipeline, merged_model, raw
    _release_accelerator()
    return result


def run_ensemble_inference(
    input_dir: Path,
    output_file: Path,
    checkpoint_path: Path,
) -> None:
    """Run all components and write the exact required prediction schema."""
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint manifest: {checkpoint_path}")
    manifest = _torch_load(checkpoint_path, map_location="cpu")
    if manifest.get("format") != "dlam_tft_tcn_chronos_v1":
        raise ValueError("Unsupported checkpoint manifest format")

    resolve = _make_asset_resolver(checkpoint_path)
    history, future, forecast_index = _load_inputs(input_dir)
    device = _device()
    print(f"Inference device: {device}", flush=True)

    tcn = _predict_tcn(
        history,
        future,
        forecast_index,
        [resolve(path) for path in manifest["tcn_checkpoints"]],
        device,
    ).rename(columns={"prediction": "prediction_tcn"})
    tft = _predict_tft(
        history,
        future,
        forecast_index,
        resolve(manifest["tft_checkpoint"]),
        device,
    ).rename(columns={"prediction": "prediction_tft"})
    chronos = _predict_chronos(
        history,
        future,
        forecast_index,
        resolve(manifest["chronos_packed_dir"]),
        float(manifest["chronos_quantile"]),
        manifest["chronos_covariate_medians"],
        device,
    ).rename(columns={"prediction": "prediction_chronos"})

    aligned = (
        forecast_index[KEYS]
        .merge(tft, on=KEYS, how="left", validate="one_to_one")
        .merge(tcn, on=KEYS, how="left", validate="one_to_one")
        .merge(chronos, on=KEYS, how="left", validate="one_to_one")
    )
    component_columns = [
        "prediction_tft",
        "prediction_tcn",
        "prediction_chronos",
    ]
    weights = np.asarray(manifest["ensemble_weights"], dtype=np.float64)
    if len(weights) != 3 or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Invalid ensemble weights: {weights}")
    aligned["prediction"] = np.maximum(
        aligned[component_columns].to_numpy(np.float64) @ weights,
        0.0,
    )
    debug_component_path = os.environ.get("DLAM_COMPONENT_DEBUG_PATH")
    if debug_component_path:
        debug_path = Path(debug_component_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        aligned[KEYS + component_columns + ["prediction"]].to_csv(
            debug_path, index=False
        )
        print(f"Saved component diagnostics to {debug_path}", flush=True)
    result = aligned[KEYS + ["prediction"]]

    if result.columns.tolist() != [ID, TIME, "prediction"]:
        raise ValueError("Wrong output schema")
    if len(result) != len(forecast_index):
        raise ValueError("Wrong output row count")
    if result.duplicated(KEYS).any():
        raise ValueError("Duplicate output keys")
    if result["prediction"].isna().any() or not np.isfinite(result["prediction"]).all():
        raise ValueError("Invalid output predictions")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"Saved {len(result):,} predictions to {output_file}", flush=True)
