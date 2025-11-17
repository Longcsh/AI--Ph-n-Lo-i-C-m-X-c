# ==========================================================
# train_models/export_error_analysis.py
# Phân tích lỗi dự đoán của mô hình PhoBERT-base-v2 (phiên bản có log)
# ==========================================================

import os, json, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# ===== 1. Đường dẫn =====
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "data_balanced.csv")
MODEL_PATH = os.path.join(BASE, "models", "phobert_base_v2_cpu_final")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(REPORTS, exist_ok=True)

print("📂 Đang tải dữ liệu và mô hình PhoBERT-base-v2...")

# ===== 2. Load dữ liệu =====
df = pd.read_csv(DATA).dropna(subset=["content", "label"])
label2id = {lbl: i for i, lbl in enumerate(sorted(df["label"].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# Chọn 10% để test
test_df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"✅ Tải {len(test_df)} mẫu test để phân tích lỗi")

# ===== 3. Load model PhoBERT đã fine-tuned =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cpu")
model.to(device)
model.eval()

# ===== 4. Hàm dự đoán =====
def predict_batch(texts, batch_size=16):
    preds_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
        print(f"   🔹 Batch {i//batch_size + 1}/{len(texts)//batch_size + 1} done")
    return preds_all

# ===== 5. Thực hiện dự đoán =====
print("🤖 Đang dự đoán trên tập test...")
test_df["pred_id"] = predict_batch(test_df["content"].tolist())
test_df["pred_label"] = test_df["pred_id"].map(id2label)
test_df["true_label"] = test_df["label_id"].map(id2label)

# ===== 6. Tạo bảng lỗi =====
errors = test_df[test_df["pred_label"] != test_df["true_label"]][["content", "true_label", "pred_label"]]
errors_path = os.path.join(REPORTS, "error_analysis.csv")
errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

# ===== 7. Lưu báo cáo Markdown =====
md_path = os.path.join(REPORTS, "error_analysis.md")
rep = classification_report(test_df["true_label"], test_df["pred_label"], target_names=sorted(label2id.keys()))

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# 🔍 Phân tích lỗi PhoBERT-base-v2\n\n")
    f.write(rep + "\n\n")
    f.write(f"Tổng số mẫu test: {len(test_df)}\n")
    f.write(f"Số lỗi: {len(errors)} ({len(errors)/len(test_df)*100:.2f}%)\n\n")
    f.write("### 🧩 Top 5 lỗi tiêu biểu:\n")
    for i, row in errors.head(5).iterrows():
        f.write(f"- **Văn bản:** {row['content'][:100]}...\n")
        f.write(f"  - Thực tế: {row['true_label']}\n")
        f.write(f"  - Dự đoán: {row['pred_label']}\n\n")

# ===== 8. Kết quả =====
print("\n✅ Phân tích lỗi hoàn tất!")
print(f"📊 Tổng số mẫu test: {len(test_df)}")
print(f"❌ Số mẫu dự đoán sai: {len(errors)} ({len(errors)/len(test_df)*100:.2f}%)")
print(f"📁 CSV: {errors_path}")
print(f"📝 Markdown: {md_path}")
