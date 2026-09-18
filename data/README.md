# Local data

- `bronze/`: immutable, provider-native landing data consumed by ingestion.
- `external/`: optional download/cache area; the pipeline never reads it directly.
- `raw/`, `interim/`, `processed/`: legacy local folders, not pipeline contracts.

Bronze payloads are ignored by Git. Their versioned contracts live in
`configs/datasets/`; generated Silver and Gold outputs remain in PostgreSQL and
`artifacts/` respectively.
