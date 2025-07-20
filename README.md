# 📊 Ứng dụng phân tích và dự đoán chất lượng sản phẩm thương mại điện tử Việt Nam

## 🎯 Mô tả đồ án

Đây là một ứng dụng web đơn giản được xây dựng bằng Python (Flask), cho phép:
- Tải lên dữ liệu sản phẩm thương mại điện tử từ sàn **Tiki.vn**
- Tự động phân tích thống kê và trực quan hóa dữ liệu
- Huấn luyện mô hình học máy để **dự đoán khả năng sản phẩm được đánh giá cao**
- Diễn giải mô hình bằng SHAP để hiểu yếu tố nào ảnh hưởng đến chất lượng sản phẩm

> **Đối tượng áp dụng**: Dữ liệu sản phẩm từ Tiki hoặc các sàn TMĐT Việt Nam với các thuộc tính như giá, số đánh giá, lượt yêu thích, hình ảnh,...

---

## 🚀 Công nghệ sử dụng

- Python 3.10
- Flask – Web Framework
- Pandas – Xử lý dữ liệu
- Matplotlib & Seaborn – Trực quan hóa
- Scikit-learn – Học máy
- SHAP – Diễn giải mô hình

---

## 📂 Cấu trúc thư mục
```flask_app/
├── app.py                  # Flask App chính
├── templates/              # Giao diện HTML
│   ├── index.html
│   └── result.html
├── static/                 # Hình ảnh biểu đồ và SHAP plot
├── uploads/                # File CSV được upload
├── utils/
│   ├── analysis.py         # Phân tích thống kê, trực quan hóa
│   └── ml_model.py         # Mô hình học máy và SHAP
```

---

## 📈 Dataset mẫu (gợi ý)

Bạn cần một dataset có các cột như sau:
- `id`, `name`, `description`
- `original_price`, `price`
- `fulfillment_type`, `brand`
- `review_counts`, `rating_average`
- `favorite_count`, `pay_later`, `current_seller`
- `date_created`, `number_of_images`

> Dataset nên có ≥ 1000 dòng, dữ liệu thực tế từ sàn TMĐT Việt Nam.

---

## ⚙️ Cách cài đặt và chạy ứng dụng

### 1. Clone dự án
```
bash
git clone https://github.com/tenban/flask-tiki-analyzer.git
cd flask-tiki-analyzer
```

### 2. Cài đặt thư viện cần thiết
```
bash
pip install -r requirements.txt
```

### 3. Chạy Flask App
```
bash
python app.py
```

### 4. Mở trình duyệt tại
```
bash
http://127.0.0.1:5000/
```



