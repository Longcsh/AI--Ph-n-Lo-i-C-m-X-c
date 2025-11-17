# ==========================================================
# train_models/export_metrics.py
# Tạo bảng tổng hợp các chỉ số Accuracy, F1-macro của baseline + PhoBERT
# ==========================================================

import os, json, pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE, "reports")
MODELS = os.path.join(BASE, "models")

summary_path = os.path.join(REPORTS, "summary.json")
phobert_summary_path = os.path.join(REPORTS, "summary_phobert_base_v2.json")

rows = []

# ===== 1. Đọc dữ liệu từ summary.json (baseline) =====
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nếu có key "val_results" thì truy cập vào trong đó
    if "val_results" in data:
        data = data["val_results"]

    for model_name, info in data.items():
        # Lấy cả val_accuracy và test_accuracy nếu có
        acc = info.get("val_accuracy") or info.get("test_accuracy", 0)
        f1 = info.get("val_macro_f1") or info.get("test_macro_f1", 0)
        rows.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "F1_macro": round(f1, 4)
        })

# ===== 2. Đọc dữ liệu từ summary_phobert_base_v2.json =====
if os.path.exists(phobert_summary_path):
    with open(phobert_summary_path, "r", encoding="utf-8") as f:
        pho = json.load(f)

    rows.append({
        "Model": "PhoBERT-base-v2",
        "Accuracy": pho.get("test_accuracy", 0),
        "F1_macro": pho.get("test_macro_f1", 0)
    })

# ===== 3. Tạo DataFrame và lưu CSV =====
df = pd.DataFrame(rows)
df = df.sort_values("F1_macro", ascending=False)

csv_path = os.path.join(REPORTS, "model_metrics_comparison.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("📊 Bảng tổng hợp chỉ số:")
print(df.to_string(index=False))
print(f"\n✅ Đã lưu bảng metrics tại: {csv_path}")

# ===== 4. Vẽ biểu đồ =====
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["F1_macro"], color="#f39c12")
plt.title("So sánh F1_macro giữa các mô hình", fontsize=13)
plt.ylabel("Giá trị (%)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

png_path = os.path.join(REPORTS, "baseline_comparison.png")
plt.savefig(png_path)
plt.show()

print(f"📈 Biểu đồ đã lưu tại: {png_path}")
