import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ Thêm dòng này để in tiếng Việt không lỗi
import pandas as pd

clean_path = r"data\data_clean.csv"
balanced_path = r"data\data_balanced.csv"

# 1️ Đọc hai file
df_clean = pd.read_csv(clean_path)
df_balanced = pd.read_csv(balanced_path)

# Xóa cột 'start' nếu tồn tại
for df in [df_clean, df_balanced]:
    if 'start' in df.columns:
        df.drop(columns=['start'], inplace=True, errors='ignore')

# 2️ Hàm thống kê nhanh
def summarize(df, name):
    return {
        "Giai đoạn": name,
        "Tổng dòng": len(df),
        "Null": df['content'].isnull().sum(),
        "Trùng": df.duplicated(subset=['content']).sum(),
        "Độ dài TB câu": round(df['content'].astype(str).apply(lambda x: len(x.split())).mean(), 2),
        "Positive (%)": round(df['label'].value_counts(normalize=True).get('positive', 0) * 100, 2),
        "Negative (%)": round(df['label'].value_counts(normalize=True).get('negative', 0) * 100, 2),
        "Neutral (%)": round(df['label'].value_counts(normalize=True).get('neutral', 0) * 100, 2)
    }

# 3️ Tạo bảng so sánh
summary = [
    summarize(df_clean, "Sau làm sạch"),
    summarize(df_balanced, "Sau cân bằng")
]

compare_df = pd.DataFrame(summary)

# 4️ In ra kết quả
print("=== BẢNG SO SÁNH SAU LÀM SẠCH – SAU CÂN BẰNG ===\n")
print(compare_df.to_string(index=False))

# 5️ Lưu ra file
output_path = r"data\compare_clean_vs_balance.csv"
compare_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu bảng thống kê tại: {output_path}")
