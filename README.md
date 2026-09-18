# MatchMind AI

> Biến StatsBomb event và 360 data thành SPADL actions, định giá từng hành động bằng VAEP và giúp scout tìm ra những đóng góp mà thống kê truyền thống bỏ sót.

MatchMind AI là một dự án player scouting analytics sử dụng dữ liệu sự kiện để trả lời những câu hỏi như:

- Cầu thủ nào tạo ra nhiều giá trị nhất trên mỗi 90 phút?
- Giá trị đó đến từ tấn công, phòng ngự hay một loại hành động cụ thể?
- Cầu thủ tạo hoặc làm mất giá trị ở khu vực nào trên sân?
- Hai cầu thủ cùng vị trí khác nhau như thế nào khi đặt trong cùng một bộ lọc?

Dự án không bắt đầu từ chatbot. Nền tảng của MatchMind là dữ liệu có thể kiểm chứng, chuỗi hành động SPADL do `socceraction` tạo và kết quả VAEP tái lập được. Lớp AI về sau chỉ giải thích những kết quả đã được hệ thống tính toán.

## Trạng thái hiện tại

| Plan | Mục tiêu | Trạng thái |
|---|---|---|
| [01 — Product Scope](docs/plans/01-product-scope.md) | Xác định người dùng, giá trị và phạm vi MVP | Hoàn thành |
| [02 — Data Foundation](docs/plans/02-data-foundation.md) | Enriched StatsBomb events, lineup intervals và 360 trong PostgreSQL | Hoàn thành |
| [03 — SPADL Actions & State Features](docs/plans/03-analytics-features.md) | Chuyển event thành action và feature theo từng trạng thái | Hoàn thành |
| [04 — VAEP Modeling](docs/plans/04-vaep-modeling.md) | Train `P_score`/`P_concede`, định giá action và tính VAEP/90 | Đã lên kế hoạch |

## Sản phẩm hướng tới

MatchMind hướng tới scout và recruitment analyst cần đánh giá cầu thủ qua nhiều trận. Analyst và người hâm mộ nâng cao là nhóm người dùng phụ.

Thay vì chỉ nhìn vào bàn thắng, kiến tạo hoặc số lần chuyền thành công, MatchMind xem xét tác động của từng hành động như chuyền bóng, kéo bóng, tranh chấp, thu hồi và dứt điểm. Hệ thống đánh giá hành động đó giúp đội tiến gần hơn đến bàn thắng hay giảm nguy cơ thủng lưới, rồi tổng hợp thành điểm đóng góp của mỗi cầu thủ.

Nhờ đó, người dùng có thể tìm ra những cầu thủ đóng góp thầm lặng, hiểu điểm mạnh và điểm yếu của họ, so sánh các ứng viên cùng vị trí và kiểm tra lại những tình huống đã tạo nên kết quả. MatchMind hỗ trợ quá trình tuyển trạch, không thay thế đánh giá chuyên môn hoặc quyết định chuyển nhượng của con người.

MVP hiện chưa xử lý video, dữ liệu vị trí liên tục hoặc dữ liệu trực tiếp; sản phẩm cũng không dự báo cá cược hay tự động đưa ra quyết định chuyển nhượng.

## Tính năng của dự án

| Tính năng | Giá trị cho người dùng | Trạng thái |
|---|---|---|
| Chuẩn bị dữ liệu trận đấu | Tổng hợp và kiểm tra dữ liệu sự kiện, đội hình và thông tin không gian từ StatsBomb | Hoàn thành nền tảng |
| Định giá từng hành động | Cho biết một pha bóng tạo thêm hay làm mất giá trị cho đội | Bước phát triển tiếp theo |
| Bảng xếp hạng cầu thủ | Xếp hạng theo tổng đóng góp và mức đóng góp trên mỗi 90 phút | Đã lên kế hoạch |
| Phân tích tấn công và phòng ngự | Giúp nhận biết giá trị của cầu thủ đến từ mặt trận nào | Đã lên kế hoạch |
| So sánh cầu thủ | So sánh hai cầu thủ cùng vị trí trên một biểu đồ thống nhất | Đã lên kế hoạch |
| Bản đồ tạo giá trị | Hiển thị những khu vực cầu thủ thường tạo ra hoặc làm mất giá trị | Đã lên kế hoạch |
| Tìm kiếm và bộ lọc | Lọc theo giải đấu, vị trí, đội bóng, độ tuổi và số phút thi đấu | Đã lên kế hoạch |
| Kiểm chứng tình huống | Xem lại các hành động đóng góp tích cực hoặc tiêu cực nhất của cầu thủ | Đã lên kế hoạch |

Dataset hiện tại chỉ có World Cup 2022 nên bộ lọc giải đấu mới có một lựa chọn. Nguồn StatsBomb đang sử dụng không có ngày sinh; bộ lọc tuổi cần thêm một nguồn thông tin cầu thủ có giấy phép và được quản lý phiên bản riêng.

## Dataset

MatchMind sử dụng [StatsBomb Open Data](https://github.com/statsbomb/open-data) được khóa bằng [vaep-training-corpus-v1.json](configs/datasets/vaep-training-corpus-v1.json). Đây là manifest nguồn duy nhất cho pipeline, gồm 1.831 trận nam thuộc tám giải và mười cặp giải-mùa trong giai đoạn 2015–2024; World Cup 2022 là một selection gồm 64 trận trong corpus này.

Dữ liệu nguồn nằm trong Bronze layer tại `data/bronze/statsbomb-open-data/` và
được giữ nguyên theo định dạng của nhà cung cấp. Trước ingestion, pipeline kiểm
tra commit Git, working tree, cấu trúc thư mục, manifest, checksum và các quan hệ
raw. Xem [Bronze contract](docs/architecture/bronze-layer.md).

```powershell
python -m matchmind.corpus.validate_bronze
```

| Nội dung | Kết quả đã xác minh |
|---|---:|
| Trận đấu | 64 |
| Sự kiện | 234.637 |
| Loại sự kiện | 33 |
| Cú sút | 1.494 |
| Cú sút có xG | 1.494 |
| Source event ID trùng | 0 |

Toàn bộ `203.882` StatsBomb 360 frame của 64 trận đã được kiểm tra, liên kết với event bằng UUID và ingest vào PostgreSQL. Event-only vẫn là contract bắt buộc; 360 là enrichment tùy chọn cho feature nâng cao và không phải dữ liệu tracking liên tục. Việc sử dụng dữ liệu tuân theo giấy phép và yêu cầu attribution của StatsBomb.

## Data foundation đã xây dựng

```mermaid
flowchart LR
    A[Bronze: StatsBomb matches/events/lineups/360] --> B[Raw Reader]
    B --> R[Raw StatsBomb Validator]
    R --> C[Normalizer]
    C --> D[Canonical Validator]
    D -->|event hợp lệ| E[(events)]
    D -->|lineup hợp lệ| H[(player_match_intervals)]
    D -->|360 hợp lệ| I[(event_360)]
    D -->|không hợp lệ| F[(invalid_events)]
    E --> G[Reconciliation]
    H --> G
    I --> G
```

Pipeline hiện có khả năng:

- Đọc dữ liệu match, lineup và event từ nguồn JSON bất biến.
- Mapping 33 loại event sang một canonical contract thống nhất.
- Giữ source event order, possession, subtype, body part, play pattern, pressure flags, related events và raw details.
- Chuẩn hóa tọa độ StatsBomb từ sân `120 × 80` về thang `0–100`.
- Lưu `2.958` player-match position intervals để tính phút thi đấu.
- Lưu và kiểm tra `203.882` StatsBomb 360 frame; trận không có 360 vẫn sử dụng được.
- Giữ đúng period, timestamp, phút bù giờ, hiệp phụ và luân lưu.
- Kiểm tra ID, kiểu dữ liệu, tọa độ, quan hệ team/player và quy định null.
- Lưu record lỗi cùng nguyên nhân vào `quarantine.invalid_events`.
- Upsert theo `(source, source_event_id)` để chạy lại mà không tạo dữ liệu trùng.
- Ghi lịch sử mỗi lần chạy vào `meta.ingestion_runs` và đối soát toàn bộ record.

Trạng thái dữ liệu sau khi ingest và enrichment đầy đủ:

```text
raw events      = 234637
rejected        = 0
events in DB    = 234637
lineup intervals= 2958
360 frames      = 203882
reconciled      = True
```

`accepted` và `deduplicated` là số liệu theo từng lần chạy nên thay đổi khi pipeline được chạy lại. Các số phía trên là trạng thái cuối trong database; khóa nguồn và cơ chế upsert bảo đảm không tạo event trùng.

## Canonical database

Các bảng PostgreSQL chính:

| Bảng | Vai trò |
|---|---|
| `silver.matches` | Thông tin trận đấu và tỷ số |
| `silver.teams` | Danh mục đội bóng |
| `silver.players` | Danh mục cầu thủ |
| `silver.events` | Event đã được chuẩn hóa và kiểm tra |
| `silver.player_match_intervals` | Khoảng vị trí/thi đấu của cầu thủ theo trận |
| `silver.event_360` | Visible area và freeze-frame liên kết theo event UUID |
| `quarantine.invalid_events` | Record bị từ chối cùng lý do |
| `meta.ingestion_runs` | Phiên bản, trạng thái và số liệu mỗi lần ingest |
| `meta.analytics_runs` | Phiên bản, lineage và trạng thái mỗi lần build analytics |
| `gold.analytics_actions` | Chuỗi SPADL action đã chuẩn bị cho phân tích |
| `gold.analytics_action_features` | Feature VAEP point-in-time theo từng action |

Chi tiết mapping StatsBomb → canonical nằm trong [data dictionary](docs/data_dictionary/statsbomb.md). Schema nền tảng nằm trong [migration 001](infra/db/migrations/001_data_foundation.up.sql), còn enrichment VAEP/360 nằm trong [migration 002](infra/db/migrations/002_vaep_event_enrichment.up.sql).

## Chạy project

Yêu cầu: Python 3.12 và Docker Desktop.

Sau khi cài dependency, khởi động PostgreSQL và áp dụng migration, có thể chạy
toàn bộ ingestion → features → model preparation → baseline training bằng:

```powershell
python -m matchmind.pipelines.run_all
```

### 1. Cài dependency

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

### 2. Khởi động PostgreSQL và pgAdmin

```powershell
docker compose up -d
docker compose ps
```

- PostgreSQL: `localhost:5433`
- pgAdmin: [http://localhost:5050](http://localhost:5050)
- Có thể thay đổi cấu hình bằng các biến trong [.env.example](.env.example).

### 3. Tạo schema ở lần chạy đầu tiên

```powershell
Get-Content -Raw infra/db/migrations/001_data_foundation.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/002_vaep_event_enrichment.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/003_spadl_analytics.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/004_medallion_silver.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/005_medallion_gold.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

### 4. Ingest toàn bộ corpus

```powershell
python -m matchmind.ingestion.run
```

Có thể chạy lại lệnh này an toàn. Event đã tồn tại sẽ được nhận diện là deduplicated thay vì được chèn thêm.
Manifest, checksum và source files của corpus được kiểm tra tự động trước khi ingest.

### 5. Chạy test

```powershell
python -m unittest discover -s tests -v
```

## SPADL analytics

Tạo baseline actions/features cho toàn bộ dataset:

```powershell
python -m matchmind.vaep_features.run
```

Bật feature contract StatsBomb 360 riêng:

```powershell
python -m matchmind.vaep_features.run --include-360
```

Output được lưu trong `meta.analytics_runs`, `gold.analytics_actions`,
`gold.analytics_action_features` và `artifacts/features/plan03/`.

## VAEP — bước kế tiếp

Plan 03 dùng `socceraction==1.5.3` chuyển enriched event thành chuỗi SPADL chuẩn,
với mỗi dòng đại diện cho:

```text
match × action
```

Mỗi state VAEP gồm đúng ba action: action hiện tại (`a0`) và hai action
trước (`a1`, `a2`). Feature baseline dùng bộ transformer mặc định của
socceraction; feature 360 vẫn là enrichment tùy chọn. Không feature nào nhìn
action tương lai.

Đầu ra chính là bảng/file `actions` và `action_features`. Plan 04 dùng chúng để train hai model XGBoost độc lập:

```text
P_score(state_i)   = P(đội thực hiện action ghi bàn trong 10 action tiếp theo)
P_concede(state_i) = P(đội thực hiện action thủng lưới trong 10 action tiếp theo)
```

Giá trị của action `a_i` được tính từ thay đổi xác suất trước và sau action:

```text
VAEP(a_i) = [P_score(after_i) - P_score(before_i)]
          + [P_concede(before_i) - P_concede(after_i)]
```

VAEP sau đó được cộng theo cầu thủ và chuẩn hóa trên 90 phút để tạo bảng xếp hạng scouting. StatsBomb 360 bổ sung context không gian khi có; action không có 360 vẫn được giữ trong baseline.

## Cấu trúc chính

```text
apps/                            Backend và frontend deploy độc lập
matchmind/                       Python package dùng chung
├── corpus/                      Corpus manifest và match metadata
├── ingestion/                   Normalize, validate và ghi PostgreSQL
├── spadl/                       Đọc canonical events và tạo SPADL actions
├── vaep_features/               Action states và VAEP feature engineering
├── labeling_and_splitting/      Targets và chronological splits
├── model_dataset/               Join dữ liệu thành model_dataset.parquet
├── model_training/              Training và evaluation
├── ai/                          AI analyst, prompts và tools
├── pipelines/                   Các pipeline entrypoint
└── shared/                      Config, contracts và logging dùng chung

configs/datasets/                Manifest khóa phiên bản dataset
data/bronze/                     StatsBomb JSON nguyên bản, không commit vào Git
infra/db/migrations/             PostgreSQL schema và rollback
infra/pgadmin/                   Cấu hình pgAdmin
tests/                           Unit, integration và end-to-end tests
artifacts/                       Features, models, runs và reports sinh tự động
docs/                            Kiến trúc, data dictionary và kế hoạch
```

Xem luồng dữ liệu tại [docs/architecture/data-flow.md](docs/architecture/data-flow.md).

---

MatchMind đang được xây dựng theo một nguyên tắc đơn giản: **phân tích hay chỉ có giá trị khi dữ liệu phía dưới đáng tin cậy**.
