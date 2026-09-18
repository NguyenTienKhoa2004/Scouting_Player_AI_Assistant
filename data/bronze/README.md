# Bronze data

Bronze is the immutable landing layer for provider-native files. MatchMind reads
StatsBomb JSON from:

```text
data/bronze/statsbomb-open-data/
├── LICENSE.pdf
└── data/
    ├── matches/<competition_id>/<season_id>.json
    ├── events/<match_id>.json
    ├── lineups/<match_id>.json
    └── three-sixty/<match_id>.json
```

Rules:

1. Preserve provider filenames, JSON shape, record order, and values.
2. Never repair or normalize files in place; corrections belong in Silver.
3. Pin the provider revision and selected inputs in `configs/datasets/`.
4. Validate the Bronze contract before normalization or database writes.
5. Keep rejected records in quarantine with their original source payload.

Only this README is committed. The Bronze payload is local and ignored by Git.

For a clean checkout, materialize the pinned source with:

```powershell
git clone https://github.com/statsbomb/open-data.git data/bronze/statsbomb-open-data
git -C data/bronze/statsbomb-open-data checkout b0bc9f22dd77c206ddedc1d742893b3bbe64baec
```

Do not edit or pull this working tree in place. Change the pinned revision through
a new dataset-manifest version and rebuild downstream layers.

Validate Bronze independently of PostgreSQL:

```powershell
py -3.12 -m matchmind.corpus.validate_bronze
```
