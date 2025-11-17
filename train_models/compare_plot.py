# ==========================================================
# compare_plot.py
# So sánh hiệu suất giữa các mô hình baseline và PhoBERT
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== 1️⃣ Đường dẫn =====
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE, "reports")
CSV_PATH = os.path.join(REPORTS, "model_metrics_comparison.csv")

# ===== 2️⃣ Đọc dữ liệu =====
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# Loại bỏ cột thừa nếu có (phòng trường hợp summary.json thêm các key khác)
df = df[["Model", "Accuracy", "F1_macro"]]
df = df.sort_values(by="F1_macro", ascending=False)

print("📊 Dữ liệu so sánh mô hình:")
print(df)

# ===== 3️⃣ Vẽ biểu đồ =====
plt.figure(figsize=(8,5))
width = 0.35
x = range(len(df))

plt.bar(x, df["Accuracy"], width=width, label="Accuracy", color="#60a5fa")
plt.bar([i + width for i in x], df["F1_macro"], width=width, label="F1_macro", color="#f59e0b")

plt.xticks([i + width/2 for i in x], df["Model"], rotation=15)
plt.ylabel("Giá trị (%)")
plt.title("So sánh hiệu suất giữa các mô hình baseline và PhoBERT")
plt.legend()
plt.tight_layout()

# ===== 4️⃣ Lưu hình =====
save_path = os.path.join(REPORTS, "final_model_comparison.png")
plt.savefig(save_path, dpi=160)
plt.show()

print(f"✅ Đã lưu biểu đồ so sánh: {save_path}")
