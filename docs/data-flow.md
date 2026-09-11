# Luồng dữ liệu MatchMind

Project có hai entrypoint chính và được đọc theo thứ tự từ trên xuống:

```text
StatsBomb Open Data
        │
        ▼
scripts/01_ingest_statsbomb.py
        │
        ├── ingestion/reader.py
        ├── ingestion/normalizer.py
        ├── ingestion/validator.py
        └── storage/postgres/ingestion_writer.py
        │
        ▼
PostgreSQL: events + lineup + event_360
        │
        ▼
scripts/02_build_features.py
        │
        ├── storage/postgres/feature_reader.py
        ├── feature_engineering/input.py
        ├── feature_engineering/spadl_converter.py
        ├── feature_engineering/action_state.py
        ├── feature_engineering/vaep_features.py
        ├── feature_engineering/features_360.py (tùy chọn)
        └── feature_engineering/pipeline.py
        │
        ├── PostgreSQL: analytics_actions + analytics_action_features
        └── Parquet: actions.parquet + action_features.parquet
```

Mỗi feature row sử dụng đúng ba action: action hiện tại `a0` và hai action
trước đó `a1`, `a2`. Thư mục `scripts/tools/` chỉ chứa công cụ kiểm tra và
không phải entrypoint của pipeline chính.
