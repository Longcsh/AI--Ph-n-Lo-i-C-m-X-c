import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ Thêm dòng này để in tiếng Việt không lỗi

import pandas as pd
import re

# === 1️⃣ Đọc dữ liệu cân bằng
path = r"data\data_balanced.csv"
df = pd.read_csv(path)

print("=== KIỂM TRA ĐỘ TIN CẬY DỮ LIỆU ===\n")
print(f"Tổng số dòng: {len(df)}")
print(f"Các nhãn hiện có: {df['label'].unique()}\n")

# === PHẦN 1 - KIỂM TRA DÒNG RỖNG / NGẮN BẤT THƯỜNG ===
short_rows = df[df['content'].str.len() < 5]
duplicate_rows = df.duplicated(subset=['content']).sum()

print("=== 1️⃣ Kiểm tra nội dung bất thường ===")
print(f"Số dòng quá ngắn (len < 5): {len(short_rows)}")
print(f"Số dòng trùng lặp: {duplicate_rows}")

if len(short_rows) > 0:
    print("\nVí dụ vài dòng bất thường:")
    print(short_rows.head(5)['content'].tolist())

# === PHẦN 2 - KIỂM TRA ĐỘ NHẤT QUÁN NỘI DUNG - NHÃN ===
positive_words = ["đẹp", "tốt", "hài lòng", "tuyệt", "ưng", "chất lượng", "rẻ", "ổn", "thích"]
negative_words = [
    "tệ", "xấu", "kém", "thất vọng", "bực", "ghét", "dở", "chán",
    "không hài lòng", "chưa tốt", "chưa ưng", "không ổn", "không thích",
    "chưa được", "chưa đẹp", "kém chất lượng", "không đáng tiền",
    "hư", "vỡ", "lỗi", "không đúng mô tả", "giao sai", "không như hình",
    "không đáng", "quá đắt", "chậm", "lừa", "đau lòng"
]

def keyword_score(text, keywords):
    return sum(1 for w in keywords if re.search(rf"\b{w}\b", text))

pos_samples = df[df['label'] == 'positive']
neg_samples = df[df['label'] == 'negative']

pos_match_rate = pos_samples['content'].apply(lambda x: keyword_score(x, positive_words) > 0).mean() * 100
neg_match_rate = neg_samples['content'].apply(lambda x: keyword_score(x, negative_words) > 0).mean() * 100

print("\n=== 2️⃣ Kiểm tra độ nhất quán nhãn ===")
print(f"Tỉ lệ câu positive có chứa từ tích cực: {pos_match_rate:.2f}%")
print(f"Tỉ lệ câu negative có chứa từ tiêu cực: {neg_match_rate:.2f}%")

if pos_match_rate >= 70 and neg_match_rate >= 70:
    print("✅ Nhãn được gán rất nhất quán và phản ánh đúng cảm xúc.")
else:
    print("⚠️ Cần xem lại một phần nhãn, có thể tồn tại gán sai hoặc câu trung tính bị lệch nghĩa.")

# === PHẦN 3 - KIỂM TRA CHỒNG LẤN NGỮ NGHĨA GIỮA CÁC NHÃN ===
neutral_keywords = ["bình thường", "tạm ổn", "cũng được", "không tệ"]

print("\n=== 3️⃣ Kiểm tra chồng lấn giữa nhãn ===")
for w in neutral_keywords:
    found_pos = len(df[(df['label'] == 'positive') & (df['content'].str.contains(w))])
    found_neg = len(df[(df['label'] == 'negative') & (df['content'].str.contains(w))])
    found_neu = len(df[(df['label'] == 'neutral') & (df['content'].str.contains(w))])
    print(f"Từ '{w}' → POS: {found_pos}, NEG: {found_neg}, NEU: {found_neu}")

print("\n✅ Hoàn thành kiểm tra độ tin cậy dữ liệu.")
