# Pipeline entrypoints

Chạy pipeline theo thứ tự:

```powershell
py -3.12 scripts/01_ingest_statsbomb.py
py -3.12 scripts/02_build_features.py
```

Script thứ nhất đưa StatsBomb vào PostgreSQL. Script thứ hai chuyển event sang
SPADL, tạo feature VAEP, ghi kết quả vào PostgreSQL và xuất Parquet.

Các file trong `tools/` chỉ dùng để kiểm tra, profile và debug dữ liệu.
