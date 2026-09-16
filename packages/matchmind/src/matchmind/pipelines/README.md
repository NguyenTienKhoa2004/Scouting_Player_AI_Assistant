# Pipeline entrypoints

Chạy toàn bộ pipeline bằng một lệnh:

```powershell
py -3.12 -m matchmind.pipelines.run_all
```

File này dùng cố định corpus và các thư mục output mặc định của project để giữ
cách chạy đơn giản. Muốn chạy riêng một công đoạn thì dùng các module bên dưới.

Chạy pipeline theo thứ tự:

```powershell
py -3.12 -m matchmind.ingestion.run
py -3.12 -m matchmind.vaep_features.run --corpus-manifest configs/datasets/vaep-training-corpus-v1.json
py -3.12 -m matchmind.model_dataset.run
py -3.12 -m matchmind.model_training.run
```

Pipeline thứ nhất đưa StatsBomb vào PostgreSQL. Pipeline thứ hai chuyển event
sang SPADL, tạo feature VAEP và xuất Parquet. Pipeline thứ ba chuẩn bị labels,
splits và model dataset. Pipeline thứ tư train và đánh giá model VAEP.

Các file trong `scripts/tools/` chỉ dùng để kiểm tra, profile và debug dữ liệu.
