import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ Thêm dòng này để in tiếng Việt không lỗi
import pandas as pd
import re
from datetime import datetime

input_path = r"data\data.csv"           # file gốc
output_path = r"data\data_clean.csv"    # file sau khi xử lý

print("=== BẮT ĐẦU LÀM SẠCH DỮ LIỆU ===")
start = datetime.now()

# 1️⃣ Đọc dữ liệu gốc
df = pd.read_csv(input_path)
print(f"Tổng số dòng ban đầu: {len(df)}")

# 2️⃣ Xóa dòng trống và trùng
df.dropna(subset=['content', 'label'], inplace=True)
df.drop_duplicates(subset=['content'], inplace=True)

# 3️⃣ Hàm làm sạch văn bản
def clean_text(text):
    text = str(text).lower()

    # Bỏ link, tag, số điện thoại
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"\b\d{9,11}\b", "", text)

    # Bỏ ký tự đặc biệt / emoji
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", " ", text)

    # Bỏ các từ hoặc mẫu spam
    stop_patterns = [
        r"\bok(e+)?\b", r"\b(a+h+)+\b", r"\b(u+h+)+\b",
        r"\b(z+a+l+o+|ib|sỉ|lẻ|inbox|shop)\b",
        r"^[0-9\s]+$", r"^.$"
    ]
    for p in stop_patterns:
        text = re.sub(p, "", text)

    # Giảm ký tự lặp bất thường
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 4️⃣ Áp dụng làm sạch
df['content'] = df['content'].apply(clean_text)

# 5️⃣ Loại bỏ câu quá ngắn hoặc dài bất thường
df = df[df['content'].str.split().apply(len) > 2]
df = df[df['content'].str.split().apply(len) < 50]

# 6️⃣ Chuẩn hóa nhãn về dạng chữ thường đầy đủ
df['label'] = df['label'].replace({
    'POS': 'positive',
    'NEG': 'negative',
    'NEU': 'neutral'
})

# 7️⃣ Xuất file kết quả
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu file sạch tại: {output_path}")
print(f"Tổng số dòng sau làm sạch: {len(df)}")
print("⏱️ Thời gian xử lý:", datetime.now() - start)
