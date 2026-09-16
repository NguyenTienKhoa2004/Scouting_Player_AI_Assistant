# Luồng dữ liệu MatchMind

Project có bốn entrypoint chính và được đọc theo thứ tự từ trên xuống:

```text
StatsBomb Open Data
        │
        ▼
matchmind.pipelines.ingestion.run
        │
        ├── matchmind/data/ingestion/reader.py
        ├── matchmind/data/ingestion/normalizer.py
        ├── matchmind/data/ingestion/validator.py
        └── matchmind/data/storage/postgres/ingestion_writer.py
        │
        ▼
PostgreSQL: events + lineup + event_360
        │
        ▼
matchmind.pipelines.feature_building.run
        │
        ├── matchmind/data/storage/postgres/feature_reader.py
        ├── matchmind/validator/spadl_input_validator.py
        ├── matchmind/analytics/features/spadl_converter.py
        ├── matchmind/analytics/features/action_state.py
        ├── matchmind/analytics/features/vaep_features.py
        ├── matchmind/analytics/features/features_360.py (tùy chọn)
        └── matchmind/analytics/features/builder.py
        │
        ├── PostgreSQL: analytics_actions + analytics_action_features
        └── Parquet: actions.parquet + action_features.parquet
        │
        ▼
matchmind.pipelines.model_preparation.run
        │
        └── matchmind/ml/         labels, splits, model dataset
        │
        ▼
matchmind.pipelines.model_training.run
        │
        └── matchmind/ml/         training, evaluation
```

Mỗi feature row sử dụng đúng ba action: action hiện tại `a0` và hai action
trước đó `a1`, `a2`. Thư mục `scripts/tools/` chỉ chứa công cụ kiểm tra và
không phải entrypoint của pipeline chính.
