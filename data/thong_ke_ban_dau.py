import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ Thêm dòng này để in tiếng Việt không lỗi
# Đọc dữ liệu gốc
df_raw = pd.read_csv(r"data\data.csv")

# Thống kê cơ bản
total_raw = len(df_raw)
null_raw = df_raw['content'].isnull().sum()
duplicates_raw = df_raw.duplicated(subset=['content']).sum()

# Độ dài trung bình câu
df_raw['length'] = df_raw['content'].astype(str).apply(lambda x: len(x.split()))
avg_len_raw = df_raw['length'].mean()

# Tỷ lệ nhãn
label_dist_raw = df_raw['label'].value_counts()
label_percent_raw = df_raw['label'].value_counts(normalize=True) * 100

print("=== THỐNG KÊ DỮ LIỆU GỐC ===")
print(f"Tổng số dòng: {total_raw}")
print(f"Số dòng null: {null_raw}")
print(f"Số dòng trùng: {duplicates_raw}")
print(f"Độ dài trung bình câu: {avg_len_raw:.2f}")
print("\nPhân bố nhãn:")
print(label_dist_raw)
print("\nTỷ lệ phần trăm:")
print(label_percent_raw)
