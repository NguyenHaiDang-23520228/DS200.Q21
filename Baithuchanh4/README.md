# DS200_TH04 - Thực Hành 4: Spark DataFrame với Java

Repository này chứa mã nguồn giải bài tập Thực hành 4 (Lab 04) môn Big Data (DS200), sử dụng **Java** và **Apache Spark DataFrame** để phân tích dữ liệu bán hàng Fecom Inc.

## Thông tin sinh viên

- **Họ và tên:** Nguyễn Hải Đăng
- **MSSV:** 23520228
- **Môn:** DS200 - Phân tích dữ liệu lớn

---

## 📂 Cấu trúc thư mục

```text
Baithuchanh4/
├── pom.xml                          # Maven + Spark 3.5.0
├── run_all.sh                       # Script chạy tự động tất cả task
├── README.md
├── assignments.ipynb                # Đề bài
├── src/main/java/
│   ├── Bai1.java                    # Đọc CSV + infer schema
│   ├── Bai2.java                    # Tổng đơn hàng, khách hàng, seller
│   ├── Bai3.java                    # Số đơn theo quốc gia
│   ├── Bai4.java                    # Số đơn theo năm/tháng
│   ├── Bai5.java                    # Thống kê Review_Score
│   ├── Bai6.java                    # Doanh thu 2024 theo danh mục (tùy chọn)
│   ├── Bai7.java                    # SP bán chạy + rating TB theo SP (tùy chọn)
│   ├── Bai10.java                   # Xếp hạng seller (tùy chọn)
│   └── RunAll.java
├── input/
│   ├── Orders.csv
│   ├── Customer_List.csv
│   ├── Order_Items.csv
│   ├── Products.csv
│   └── Order_Reviews.csv
└── output/
    ├── bai1/ … bai5/
    ├── bai6/, bai7/, bai10/
```

---

## 🚀 Hướng dẫn chạy dự án

### Yêu cầu hệ thống

1. Java JDK 11+
2. Apache Maven
3. RAM khuyến nghị ≥ 4GB (Spark local)

```bash
java -version
mvn -version
```

### Cách 1: Chạy tự động tất cả các Task

```bash
cd Baithuchanh4
chmod +x run_all.sh
./run_all.sh
```

Hoặc dùng class `RunAll`:

```bash
mvn clean package dependency:copy-dependencies
java -Xmx4g -cp "target/Baithuchanh4-1.0-SNAPSHOT.jar:target/dependency/*" RunAll .
```

### Cách 2: Chạy thủ công từng Task

**Bước 1: Build**

```bash
cd Baithuchanh4
mvn clean package dependency:copy-dependencies
```

**Bước 2: Khai báo biến**

```bash
JAR=target/Baithuchanh4-1.0-SNAPSHOT.jar
CP="${JAR}:target/dependency/*"
INPUT=input
OUTPUT=output
```

**Bước 3: Chạy từng bài**

> Chỉ copy dòng lệnh, không copy ` ```bash ` hay ` ``` `.

```bash
java -Xmx4g -cp "$CP" Bai1 "$INPUT" "$OUTPUT/bai1"
java -Xmx4g -cp "$CP" Bai2 "$INPUT" "$OUTPUT/bai2"
java -Xmx4g -cp "$CP" Bai3 "$INPUT" "$OUTPUT/bai3"
java -Xmx4g -cp "$CP" Bai4 "$INPUT" "$OUTPUT/bai4"
java -Xmx4g -cp "$CP" Bai5 "$INPUT" "$OUTPUT/bai5"
java -Xmx4g -cp "$CP" Bai6 "$INPUT" "$OUTPUT/bai6"
java -Xmx4g -cp "$CP" Bai7 "$INPUT" "$OUTPUT/bai7"
java -Xmx4g -cp "$CP" Bai10 "$INPUT" "$OUTPUT/bai10"
```

---

## 📊 Kết quả (Output)

Kết quả ghi vào `output/baiX/` dưới dạng CSV Spark (`part-00000-....csv` + `_SUCCESS`).

Xem nhanh:

```bash
head output/bai2/part-*.csv
cat output/bai3/part-*.csv
```

Xem tất cả:

```bash
for d in bai1 bai2 bai3 bai4 bai5 bai6 bai7 bai10; do
  echo "========== $d =========="
  cat output/$d/part-*.csv
  echo
done
```

---

## Mô tả các bài

| Bài | Mục tiêu | Input chính |
|-----|----------|-------------|
| 1 | Đọc CSV, infer schema từng cột | 5 file CSV |
| 2 | Tổng số đơn hàng, khách hàng, seller | Orders, Customer_List, Order_Items |
| 3 | Số đơn theo quốc gia (giảm dần) | Orders + Customer_List |
| 4 | Số đơn theo năm (↑) / tháng (↓) | Orders |
| 5 | TB điểm review + số lượng theo mức 1–5 (lọc NULL/outlier) | Order_Reviews |
| 6 | Doanh thu 2024 theo danh mục (Price + Freight) | Order_Items + Products + Orders |
| 7 | SP bán chạy nhất + rating TB từng SP | Order_Items + Order_Reviews + Products |
| 10 | Xếp hạng seller theo doanh thu & số đơn | Order_Items |

> **Bài tùy chọn:** Đề yêu cầu chọn 3 trong câu 6–10. Dự án này làm **6, 7, 10**.

---

## Ghi chú kỹ thuật

- CSV dùng delimiter **`;`** (không phải dấu phẩy).
- Đọc file với `header=true`, `inferSchema=true`.
- Chạy Spark ở chế độ **local[*]**.
- `Review_Score` chỉ lấy giá trị **1–5**, loại NULL và ngoại lệ.
- Doanh thu = `Price + Freight_Value`.

---

## Chụp màn hình nộp bài

1. Chạy `./run_all.sh` hoặc từng lệnh `java -cp ...` ở trên.
2. Chụp terminal khi task chạy xong (`BaiX completed`).
3. Chụp nội dung `output/baiX/part-*.csv` cho từng bài.
