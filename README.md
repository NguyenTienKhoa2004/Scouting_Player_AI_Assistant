# PitchPulse

> Turning every football action into measurable value.

PitchPulse is an end-to-end football analytics and machine-learning system that measures how every on-ball action changes a team's probability of scoring or conceding. It is designed for player scouting and performance analysis: passes, carries, shots, recoveries, duels, and other actions are valued in context, then aggregated into offensive, defensive, total, and per-90 player metrics.

The repository contains the complete batch analytics core: versioned data ingestion, validation, feature engineering, model training, frozen-test evaluation, action valuation, player aggregation, PostgreSQL persistence, and reproducibility checks.

## What the project delivers

- A pinned, checksum-validated StatsBomb Open Data corpus.
- An idempotent Bronze-to-Silver ingestion pipeline for matches, events, lineups, and optional StatsBomb 360 frames.
- Deterministic conversion from canonical events to SPADL actions.
- Leakage-safe action-state features using only the current and two previous actions.
- Independent `P_score` and `P_concede` models with logistic baselines, XGBoost training, probability calibration, and a frozen test set.
- VAEP values for every eligible action.
- Player aggregates by match and competition-season, including total, offensive, defensive, and per-90 VAEP.
- Versioned Parquet artifacts and PostgreSQL Gold tables with lineage, hashes, run metadata, and quality reports.
- A reproducibility workflow that rebuilds and compares model artifacts.

## System architecture

```mermaid
flowchart LR
    A["Bronze<br/>StatsBomb JSON"] --> B["Validate, normalize,<br/>and reconcile"]
    B -->|accepted| C[("PostgreSQL Silver<br/>events, lineups, 360")]
    B -->|rejected| Q[("Quarantine")]
    C --> D["SPADL conversion"]
    D --> E["Point-in-time<br/>state features"]
    E --> F["Labels and chronological<br/>match-level splits"]
    F --> G["Logistic baselines +<br/>calibrated XGBoost"]
    G --> H["Frozen test evaluation"]
    H --> I["Action VAEP"]
    I --> J["Player VAEP<br/>and VAEP/90"]
    E --> K[("PostgreSQL Gold<br/>SPADL + features")]
    F --> L[("Parquet model dataset")]
    J --> M[("PostgreSQL Gold<br/>labels, values, aggregates")]
```

The processing path is deliberately separated from the application layer. `pitchpulse/` owns reusable analytics and ML logic; `apps/backend/` and `apps/frontend/` are reserved boundaries for a future API and dashboard and are not required to run the pipeline.

## Data engineering

### Source corpus

The default corpus is declared in [`configs/datasets/vaep-training-corpus-v1.json`](configs/datasets/vaep-training-corpus-v1.json) and pins StatsBomb Open Data to commit `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`.

| Coverage | Value |
|---|---:|
| Matches | 1,831 |
| Competitions | 8 |
| Competition-seasons | 10 |
| Date range | 2015-08-07 to 2024-07-15 |
| Gender scope | Men's competitions |
| Required inputs | Matches, events, lineups |
| Optional enrichment | StatsBomb 360 |

The corpus includes Premier League, Ligue 1, La Liga, Serie A, FIFA World Cup, UEFA Euro, African Cup of Nations, and Copa America selections. Provider data stays immutable under `data/bronze/` and is excluded from Git.

### Medallion pipeline

**Bronze** is the provider-native boundary. The loader verifies the pinned Git revision, manifest schema, paths, file hashes, declared match counts, required input families, and cross-record relationships before database access.

**Silver** is the validated PostgreSQL representation. The ingestion service:

- normalizes StatsBomb coordinates and event fields into typed canonical records;
- preserves source IDs, order, timestamps, periods, possession, play pattern, subtype, body part, related events, and raw details;
- derives player participation and position intervals from lineups;
- links optional 360 freeze frames and visible areas by event UUID;
- quarantines invalid records with their rejection reason;
- performs idempotent upserts and reconciles every ingestion run.

**Gold** contains versioned analytics and ML outputs: SPADL actions, point-in-time feature rows, target labels, calibrated action values, and player VAEP aggregates.

### PostgreSQL schemas

| Schema | Main objects | Purpose |
|---|---|---|
| `meta` | `ingestion_runs`, `analytics_runs`, `vaep_model_runs` | Execution status, lineage, versions, counts, metrics, and artifact hashes |
| `quarantine` | `invalid_events` | Rejected source records and validation reasons |
| `silver` | `matches`, `teams`, `players`, `events`, `player_match_intervals`, `event_360` | Canonical football data |
| `gold` | `spadl_actions`, `spadl_action_features`, `vaep_action_labels`, `action_values`, `player_vaep` | Analytics-ready and model-derived data |

Database migrations are reversible and live in [`infra/db/migrations/`](infra/db/migrations/). See the [data dictionary](docs/data_dictionary/statsbomb.md) for the source-to-canonical mapping.

## Machine learning

### Feature engineering

StatsBomb events are converted to `socceraction==1.5.3` SPADL actions on a `105 x 68` pitch. Each model row represents one action state and contains features for exactly three actions:

```text
a0 = current action
a1 = previous action
a2 = action before a1
```

The baseline feature contract contains 568 allowlisted features. It uses only information available at the current action, preventing future-action leakage. StatsBomb 360 features are available through a separate optional contract; matches or actions without 360 data remain valid for the event-data baseline.

### Targets and splits

For state `i`, the two binary targets are:

```text
scores_i   = the acting team scores within the next 10 actions
concedes_i = the acting team concedes within the next 10 actions
```

Complete matches are ordered chronologically and assigned once to a `70% / 15% / 15%` train, validation, and test split. The split has no random seed and never separates actions from the same match. The test set remains untouched until the selected and calibrated models are frozen.

### Training and evaluation

The pipeline trains class-weighted logistic SGD baselines, evaluates several XGBoost candidates, fits sigmoid probability calibrators on a dedicated validation phase, and compares the selected models with the baselines. The frozen bundle is then evaluated once on the held-out test set using PR AUC, ROC AUC, Brier score, log loss, precision, recall, calibration curves, latency, group diagnostics, and football sanity checks.

The committed evaluation policy is defined in [`configs/models/vaep-test-evaluation-v1.json`](configs/models/vaep-test-evaluation-v1.json).

### VAEP valuation

The calibrated models estimate scoring and conceding probabilities before and after each action:

```text
offensive_value(a_i) = P_score(after_i) - P_score(before_i)
defensive_value(a_i) = P_concede(before_i) - P_concede(after_i)

VAEP(a_i) = offensive_value(a_i) + defensive_value(a_i)
```

Action values are reconciled and then aggregated by player, team, position, action type, match, and competition-season. Player rankings default to a minimum of 450 minutes and expose:

- total VAEP;
- offensive VAEP;
- defensive VAEP;
- action count and match count;
- minutes played;
- VAEP per 90 minutes.

## Verified pipeline output

The checked-in run manifests record the following completed output for the full corpus:

| Artifact | Verified result |
|---|---:|
| SPADL actions / labels | 3,683,524 |
| Eligible model rows | 3,683,283 |
| Model features | 568 |
| Train matches | 1,281 |
| Validation matches | 275 |
| Test matches | 275 |
| Action VAEP rows | 3,683,283 |
| Players | 4,050 |
| Player aggregate rows | 559,408 |
| PostgreSQL modeling run | Persisted and reconciled |

Frozen-test metrics for the calibrated XGBoost bundle:

| Target | PR AUC | ROC AUC | Brier score |
|---|---:|---:|---:|
| Scores | 0.2063 | 0.8165 | 0.00825 |
| Concedes | 0.0656 | 0.8205 | 0.00180 |

Both targets passed the configured quality gate and beat their logistic baselines during validation. Full metrics, calibration diagnostics, error analysis, and sanity checks are stored with the model artifacts under `artifacts/models/`.

## Quick start

### Requirements

- Python 3.12+
- Docker Desktop with Docker Compose
- Git

### 1. Create the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Materialize the pinned Bronze data

```powershell
git clone https://github.com/statsbomb/open-data.git data/bronze/statsbomb-open-data
git -C data/bronze/statsbomb-open-data checkout b0bc9f22dd77c206ddedc1d742893b3bbe64baec
python -m pitchpulse.corpus.validate_bronze
```

StatsBomb Open Data is subject to its own license and attribution requirements. Keep its `LICENSE.pdf` with the local Bronze copy.

### 3. Start PostgreSQL and apply migrations

```powershell
docker compose up -d
docker compose ps

Get-ChildItem infra/db/migrations/*.up.sql |
  Sort-Object Name |
  ForEach-Object {
    Get-Content -Raw $_.FullName |
      docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
    if ($LASTEXITCODE -ne 0) { throw "Migration failed: $($_.Name)" }
  }
```

Default local services:

- PostgreSQL: `localhost:5433`
- pgAdmin: [http://localhost:5050](http://localhost:5050)
- Database URL: `postgresql://pitchpulse:1234567@localhost:5433/pitchpulse`

Change local credentials and ports in `.env` before using the stack outside a disposable development environment.

### 4. Run the complete pipeline

```powershell
python -m pitchpulse.pipelines.run_all
```

This command executes ingestion, SPADL and feature generation, feature registration, label and dataset creation, model training, frozen-test evaluation, action valuation, player aggregation, and PostgreSQL persistence in dependency order.

The full corpus produces large Parquet artifacts and millions of database rows. Feature construction defaults to five matches per batch so it can run on a machine with approximately 8 GB of memory.

## Run individual stages

```powershell
# Bronze -> PostgreSQL Silver
python -m pitchpulse.ingestion.run

# Silver -> SPADL actions and point-in-time features
python -m pitchpulse.vaep_features.run
python -m pitchpulse.vaep_features.register_corpus

# Labels, chronological splits, and model dataset
python -m pitchpulse.model_dataset.run

# Logistic baselines, XGBoost selection, and calibration
python -m pitchpulse.model_training.run

# One-time evaluation of the frozen model bundle
python -m pitchpulse.model_training.run_test_evaluation

# Calibrated inference and action-level VAEP
python -m pitchpulse.model_training.run_valuation

# Player-level aggregation; change the ranking threshold if needed
python -m pitchpulse.player_vaep.run --minimum-minutes 450

# Persist completed labels, action values, and player aggregates
python -m pitchpulse.model_training.run_persistence
```

To enable the separate StatsBomb 360 feature contract:

```powershell
python -m pitchpulse.vaep_features.run --include-360
```

To independently rebuild the modeling stages and compare artifact hashes:

```powershell
python -m pitchpulse.reproducibility.run
```

## Testing

```powershell
python -m unittest discover -s tests -v
```

The suite covers raw and canonical validation, idempotent ingestion, SPADL conversion, feature contracts, target generation, chronological splits, model datasets, training and evaluation components, VAEP calculations, player aggregation, PostgreSQL integration, pipeline orchestration, and reproducibility verification.

## Repository structure

```text
apps/                         API and dashboard application boundaries
configs/datasets/             Versioned corpus manifests
configs/models/               Model evaluation policies
data/bronze/                  Immutable provider data; ignored by Git
docs/architecture/            Data-layer and system architecture
docs/data_dictionary/         StatsBomb-to-canonical field definitions
infra/db/migrations/          PostgreSQL migrations and rollbacks
infra/pgadmin/                Local pgAdmin configuration
pitchpulse/corpus/             Corpus loading and source validation
pitchpulse/ingestion/          Raw validation, normalization, and Silver writes
pitchpulse/spadl/              Canonical event-to-action conversion
pitchpulse/vaep_features/      Action states, features, artifacts, and registration
pitchpulse/labeling_and_splitting/  Future targets and chronological splits
pitchpulse/model_dataset/      Leakage-safe model dataset assembly
pitchpulse/model_training/     Baselines, XGBoost, calibration, evaluation, valuation
pitchpulse/player_vaep/        Minutes calculation and player aggregation
pitchpulse/reproducibility/    Independent rebuild and artifact comparison
pitchpulse/pipelines/          End-to-end orchestration
artifacts/                    Generated features, models, values, and reports
tests/                        Unit, integration, and end-to-end test areas
```

## Current scope

PitchPulse is currently a production-style offline data and ML pipeline. The FastAPI backend, interactive scouting dashboard, and conversational analyst are not implemented in this repository yet. The analytics use event data and optional event-linked 360 snapshots, not continuous player tracking or video. VAEP supports scouting decisions; it does not replace contextual video review or human recruitment judgment.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Data flow](docs/architecture/data-flow.md)
- [Bronze contract](docs/architecture/bronze-layer.md)
- [Silver contract](docs/architecture/silver-layer.md)
- [Gold contract](docs/architecture/gold-layer.md)
- [Database migrations](infra/db/migrations/README.md)
- [Pipeline entrypoints](pitchpulse/pipelines/README.md)

---

PitchPulse is built around one principle: football analytics is useful only when its data, features, models, and outputs can be traced and reproduced.
