# Luồng dữ liệu MatchMind

Code được tổ chức theo đúng thứ tự các stage của pipeline:

```text
Bronze: immutable StatsBomb Open Data JSON
        │
        ▼
matchmind/corpus/
        └── khóa và kiểm tra tập trận từ manifest
        │
        ▼
matchmind/ingestion/
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
matchmind/spadl/
        ├── input_reader.py
        ├── input_validator.py
        └── converter.py
        │
        ▼
matchmind/vaep_features/
        ├── action_state.py
        ├── feature_builder.py
        ├── features_360.py (tùy chọn)
        ├── builder.py
        └── artifacts.py
        │
        ▼
PostgreSQL Gold: SPADL actions + point-in-time action features
        │
        ▼
Parquet: actions.parquet + action_features.parquet
        │
        ▼
matchmind/labeling_and_splitting/
        ├── targets.py + label_artifacts.py
        └── splits.py + split_artifacts.py
        │
        ▼
matchmind/model_dataset/
        ├── builder.py
        ├── artifacts.py
        └── finalize.py
        │
        ▼
model_dataset.parquet
        │
        ▼
matchmind/model_training/
```

Các entrypoint chạy tuần tự:

```text
matchmind.ingestion.run
matchmind.vaep_features.run
matchmind.model_dataset.run
matchmind.model_training.run
```

Mỗi feature row dùng đúng ba action: action hiện tại `a0` và hai action trước đó
`a1`, `a2`.
