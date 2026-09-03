# Final TFT-TCN-Chronos submission

This archive contains the final full-data ensemble for the DLAM SS26 bonus
project. Inference is offline and uses three direct-336 TCN seeds, one Temporal
Fusion Transformer, and a locally packaged Chronos-2 base model with the final
LoRA adapter.

The final non-negative convex weights and selected Chronos quantile are stored
in `checkpoint.pt`. They were selected with a chronological week-1 holdout and
accepted only after an untouched week-2 audit. Chronos is explicitly placed in
evaluation mode and its LoRA weights are merged before prediction.

## Required command

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

For local validation, `input_dir` must contain `train.csv`,
`validation_input.csv`, and `forecast_index_validation.csv`. During private
evaluation it may instead contain `test_input.csv`, `forecast_index_test.csv`,
and `metadata.json`.

The script automatically uses CUDA when available and otherwise falls back to
CPU. No model or data is downloaded during inference.

## Expected output

The output contains exactly:

```text
series_id,timestamp,prediction
```

with one row for every row in the supplied forecast index.
