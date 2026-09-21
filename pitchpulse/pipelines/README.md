# Pipeline entrypoints

Chạy toàn bộ pipeline bằng một lệnh:

```powershell
python -m pitchpulse.pipelines.run_all
```

File này dùng cố định corpus và các thư mục output mặc định của project để giữ
cách chạy đơn giản. Muốn chạy riêng một công đoạn thì dùng các module bên dưới.

Chạy pipeline theo thứ tự:

```powershell
python -m pitchpulse.ingestion.run
python -m pitchpulse.vaep_features.run --corpus-manifest configs/datasets/vaep-training-corpus-v1.json
python -m pitchpulse.vaep_features.register_corpus
python -m pitchpulse.model_dataset.run
python -m pitchpulse.model_training.run
python -m pitchpulse.model_training.run_test_evaluation
python -m pitchpulse.model_training.run_valuation
python -m pitchpulse.player_vaep.run
python -m pitchpulse.model_training.run_persistence
python -m pitchpulse.reproducibility.run
```

The final reproducibility command performs a clean Plan 04 rerun in
`artifacts/reproducibility/plan04`, compares every required Parquet and model
artifact byte-for-byte, compares JSON reports after removing path and timing
fields, then records `reproducibility_report.json` and its SHA-256 in the
promoted run's `training_manifest.json`.

The registration step creates or reuses the real Plan 03 analytics run for the
same 1,831-match corpus. Wide model features remain in Parquet; exact action
keys are copied to `gold.spadl_actions` for Task 13 foreign keys. The old
64-match run is rejected by the lineage and count checks.

Pipeline thứ nhất đưa StatsBomb vào PostgreSQL. Pipeline thứ hai chuyển event
sang SPADL, tạo feature VAEP và xuất Parquet. Pipeline thứ ba chuẩn bị labels,
splits và model dataset. Các bước tiếp theo train model, đánh giá test đã đóng
băng, tính VAEP từng action và tổng hợp VAEP cầu thủ.
