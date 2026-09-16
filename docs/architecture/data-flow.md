# Luồng dữ liệu MatchMind

Code được tổ chức theo đúng thứ tự các stage của pipeline:

```text
StatsBomb Open Data
        │
        ▼
matchmind/corpus/
        └── khóa và kiểm tra tập trận từ manifest
        │
        ▼
matchmind/ingestion/
        ├── reader.py
        ├── normalizer.py
        ├── validator.py
        ├── service.py
        └── postgres_writer.py
        │
        ▼
PostgreSQL: events + lineup + event_360
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
Parquet: actions.parquet + action_features.parquet
        │
        ▼
matchmind/labeling_and_splitting/
        ├── targets.py
        └── splits.py
        │
        ▼
matchmind/model_dataset/
        ├── builder.py
        ├── preparation.py
        └── artifacts.py
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
`a1`, `a2`. Thư mục `scripts/tools/` chỉ chứa công cụ kiểm tra và debug, không
phải entrypoint của pipeline chính.
