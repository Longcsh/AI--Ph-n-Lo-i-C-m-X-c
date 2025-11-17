# 🔍 Phân tích lỗi PhoBERT-base-v2

              precision    recall  f1-score   support

    negative       0.93      0.96      0.94       574
     neutral       0.94      0.90      0.92       622
    positive       0.94      0.95      0.95       604

    accuracy                           0.94      1800
   macro avg       0.94      0.94      0.94      1800
weighted avg       0.94      0.94      0.94      1800


Tổng số mẫu test: 1800
Số lỗi: 114 (6.33%)

### 🧩 Top 5 lỗi tiêu biểu:
- **Văn bản:** k nhu mong đợi k nghĩ form lớn như vậy...
  - Thực tế: negative
  - Dự đoán: neutral

- **Văn bản:** chất lượng sản phẩm chưa thể đánh giá vì chưa dùng đóng gói sản phẩm rất đẹp và chắc chắn thời gian ...
  - Thực tế: neutral
  - Dự đoán: positive

- **Văn bản:** mỗi tội mất moc khoá ngay mặt...
  - Thực tế: neutral
  - Dự đoán: negative

- **Văn bản:** chất lượng sản phẩm tuyệt vời giáo hàng khá nhanh có điều hộp bị móp nhiều...
  - Thực tế: neutral
  - Dự đoán: positive

- **Văn bản:** điểm trừ duy nhất là áo adidas mà tem áo lả mango nhưng không ảnh hưởng gì hết...
  - Thực tế: positive
  - Dự đoán: neutral

