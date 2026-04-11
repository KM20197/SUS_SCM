# Reproducibility

## Determinism and pairing

The experiment design uses paired replications. Preserve each `runs/<run_id>/` folder intact.

## Expected outputs

- raw_replications.csv
- summary_means_ci.csv
- wilcoxon_holm.csv
- run_manifest.json
- run_fingerprint.json

## Post-processing

Write derived tables to separate `runs/postproc_<timestamp>/` folders and archive them for audit.
