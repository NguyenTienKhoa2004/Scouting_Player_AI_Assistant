# Luồng dữ liệu PitchPulse

Code được tổ chức theo đúng thứ tự các stage của pipeline:

```text
Bronze: immutable StatsBomb Open Data JSON
        │
        ▼
pitchpulse/corpus/
        └── khóa và kiểm tra tập trận từ manifest
        │
        ▼
pitchpulse/ingestion/
        ├── reader.py
        ├── raw_validation/
        ├── normalizer.py
        ├── validator.py
        ├── service.py
        └── postgres_writer.py
        │
        ▼
PostgreSQL Silver: events + lineup + event_360
        │
        ▼
pitchpulse/spadl/
        ├── input_reader.py
        ├── input_validator.py
        └── converter.py
        │
        ▼
pitchpulse/vaep_features/
        ├── action_state.py
        ├── feature_builder.py
        ├── features_360.py (tùy chọn)
        ├── feature_pipeline.py
        └── artifacts.py
        │
        ▼
PostgreSQL Gold: SPADL actions + point-in-time action features
        │
        ▼
Parquet: actions.parquet + action_features.parquet
        │
        ▼
pitchpulse/labeling_and_splitting/
        ├── targets.py + label_artifacts.py
        └── splits.py + split_artifacts.py
        │
        ▼
pitchpulse/model_dataset/
        ├── artifacts.py
        ├── feature_allowlist.py
        └── finalize.py
        │
        ▼
model_dataset.parquet
        │
        ▼
pitchpulse/model_training/
        │
        ▼
action_values.parquet
        │
        ▼
pitchpulse/player_vaep/
        ├── calculations.py
        ├── sources.py
        ├── artifacts.py
        └── run.py
        │
        ▼
player_vaep.parquet
        │
        ▼
PostgreSQL Gold: VAEP labels + action values + player aggregates
```

Các entrypoint chạy tuần tự:

```text
pitchpulse.ingestion.run
pitchpulse.vaep_features.run
pitchpulse.vaep_features.register_corpus
pitchpulse.model_dataset.run
pitchpulse.model_training.run
pitchpulse.model_training.run_test_evaluation
pitchpulse.model_training.run_valuation
pitchpulse.player_vaep.run
pitchpulse.model_training.run_persistence
```

Mỗi feature row dùng đúng ba action: action hiện tại `a0` và hai action trước đó
`a1`, `a2`.
