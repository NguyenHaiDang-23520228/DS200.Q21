# DS200 - Bài Thực Hành 5: Hệ thống đếm người qua Camera (Distributed TCP)

## Thông tin sinh viên

- **Họ và tên:** Nguyễn Hải Đăng
- **MSSV:** 23520228
- **Môn:** DS200 - Phân tích dữ liệu lớn

---

## Mô tả hệ thống

Hệ thống gồm **3 server phân tán**, giao tiếp qua **TCP + JSON** (theo mẫu `tcp_example.py`):

```
Camera/Webcam/Video
        |
        v
+----------------+      TCP:6100       +-------------------+      TCP:6200       +----------------+
| Camera Server  | -----------------> | Processing Server | -----------------> | Storage Server |
| camera_server  |   frame (base64)   | processing_server |  detection result  | storage_server |
+----------------+                    +-------------------+                      +----------------+
                                              |                                           |
                                       OpenCV HOG                               output/results/
                                       person detector                           JSON + NDJSON
```

### Vai trò từng server

| Server | File | Chức năng |
|--------|------|-----------|
| Camera | `camera_server.py` | Nhận khung hình từ webcam/video/ảnh, gửi đến server xử lý |
| Processing | `processing_server.py` | Nhận diện người (HOG), trả bounding box + số lượng người |
| Storage | `storage_server.py` | Lưu kết quả phát hiện, tạo file tổng hợp |

### Liên hệ công nghệ dữ liệu lớn

- Kiến trúc **microservices phân tán** (tách luồng thu, xử lý, lưu trữ)
- Mô hình **stream processing**: camera stream -> detector -> data lake
- Lưu trữ dạng **NDJSON** (`detections.ndjson`) để batch analytics
- Script `analyze_results.py` mô phỏng **batch processing** trên dữ liệu đã lưu

---

## Cấu trúc thư mục

```text
Baithuchanh5/
├── tcp_example.py           # Mẫu TCP thầy gửi
├── config.py                # Cấu hình host/port
├── common.py                # Hàm TCP + JSON dùng chung
├── camera_server.py         # Server 1: thu khung hình
├── processing_server.py     # Server 2: nhận diện người
├── storage_server.py        # Server 3: lưu kết quả
├── prepare_input.py         # Tạo video/ảnh mẫu
├── analyze_results.py       # Phân tích batch kết quả
├── run_system.sh            # Chạy tự động cả hệ thống
├── requirements.txt
├── input/
│   ├── sample.mp4
│   └── frames/
├── output/results/
│   ├── frame_0001.json
│   ├── detections.ndjson
│   └── summary.json
└── logs/
```

---

## Cài đặt

```bash
cd Baithuchanh5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 prepare_input.py
```

---

## Cách 1: Chạy tự động toàn hệ thống

```bash
chmod +x run_system.sh
./run_system.sh
```

Script sẽ:
1. Tạo dữ liệu mẫu (`input/sample.mp4`)
2. Khởi động Storage Server (port 6200)
3. Khởi động Processing Server (port 6100)
4. Chạy Camera Server gửi 10 khung hình
5. Ghi kết quả vào `output/results/`

---

## Cách 2: Chạy thủ công từng server

Mở **3 terminal** trong thư mục `Baithuchanh5`:

**Terminal 1 — Storage Server**
```bash
python3 storage_server.py
```

**Terminal 2 — Processing Server**
```bash
python3 processing_server.py
```

**Terminal 3 — Camera Server**
```bash
python3 prepare_input.py
python3 camera_server.py
```

**Phân tích batch (tuỳ chọn):**
```bash
python3 analyze_results.py
```

---

## Demo realtime + bounding box (chứng minh hệ thống hoạt động)

### Cách A — Video có người + cửa sổ preview (khuyến nghị chụp màn hình)

Mở **3 terminal**:

```bash
# Terminal 1
python3 storage_server.py

# Terminal 2
python3 processing_server.py

# Terminal 3 — dùng video của bạn hoặc sample.mp4
python3 camera_server.py --source video --video input/sample.mp4 --show --max-frames 10
```

Hoặc chạy nhanh:
```bash
chmod +x run_demo.sh
./run_demo.sh
```

Kết quả:
- Cửa sổ **Real-time People Counting** hiện khung xanh (bounding box)
- Ảnh có box lưu tại `output/annotated/frame_XXXX.jpg`
- JSON lưu tại `output/results/frame_XXXX.json`

Tạo lại ảnh có box từ JSON:
```bash
python3 visualize_results.py
```

### Cách B — Webcam realtime (chạy thật trước lớp)

```bash
python3 camera_server.py --source webcam --realtime --show
```

Nhấn **Q** để dừng.

### Cách C — Dùng video tự quay / tải về

1. Copy video vào `input/my_video.mp4`
2. Chạy:
```bash
python3 camera_server.py --source video --video input/my_video.mp4 --show --realtime
```

> Video có **người đi lại** sẽ detect tốt hơn `sample.mp4` (video mẫu chỉ có hình vẽ).

---

## Làm sao chứng minh đây là 3 server thật?

| Cách chứng minh | Làm gì | Chụp gì |
|-----------------|--------|---------|
| **3 process riêng** | Mở 3 terminal, mỗi cái chạy 1 server | Screenshot 3 terminal cùng lúc |
| **Port TCP** | `lsof -i :6100` và `lsof -i :6200` | Thấy process Python listen port |
| **Kill test** | Tắt Processing Server, chạy Camera → báo `Connection refused` | Chứng minh Camera phụ thuộc server xử lý |
| **Log từng tầng** | `[Camera] Sent` → `[Processing] person_count` → `[Storage] Saved` | Screenshot chuỗi log |
| **JSON + bounding box** | `frame_0001.json` có `bounding_boxes` + ảnh `output/annotated/` | Chứng minh pipeline xử lý có kết quả |

Ví dụ kiểm tra port:
```bash
lsof -i :6100   # Processing server
lsof -i :6200   # Storage server
```

---

## Xem kết quả

```bash
cat output/results/summary.json
cat output/results/frame_0001.json
head output/results/detections.ndjson
```

Ví dụ `summary.json`:

```json
{
  "total_frames": 10,
  "total_persons_detected": 0,
  "average_person_count": 0.0,
  "max_person_count": 0
}
```

Mỗi file `frame_XXXX.json` chứa:
- `person_count`
- `bounding_boxes` (x, y, width, height)
- `processing_time_ms`

---

## Cấu hình

Chỉnh trong `config.py`:

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `processing_port` | 6100 | Camera -> Processing |
| `storage_port` | 6200 | Processing -> Storage |
| `max_frames` | 10 | Số khung hình gửi mỗi lần chạy |
| `camera_index` | 0 | Webcam (ưu tiên nếu có) |

Thứ tự nguồn khung hình: **video mẫu -> ảnh -> webcam -> synthetic**.

---

## Chụp màn hình nộp bài

1. Terminal 3 server đang chạy / log `person_count`
2. Nội dung `output/results/summary.json`
3. Một file `frame_XXXX.json` có `bounding_boxes`
4. Kết quả `python3 analyze_results.py`

---

## Giao thức TCP (theo tcp_example.py)

Payload JSON kết thúc bằng `\n`:

**Camera -> Processing**
```json
{
  "type": "frame",
  "frame_id": 1,
  "timestamp": "2026-05-29T10:00:00",
  "width": 640,
  "height": 480,
  "image_base64": "..."
}
```

**Processing -> Storage**
```json
{
  "type": "detection_result",
  "frame_id": 1,
  "person_count": 2,
  "bounding_boxes": [{"x": 10, "y": 20, "width": 80, "height": 180}],
  "processing_time_ms": 120
}
```
