# Local data

- `external/`: immutable third-party sources such as StatsBomb Open Data.
- `raw/`: snapshots received from ingestion sources.
- `interim/`: partially transformed working data.
- `processed/`: validated, analysis-ready datasets.

Dataset contents are ignored by Git. Versioned manifests live in
`configs/datasets/`.
