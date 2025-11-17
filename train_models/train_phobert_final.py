# ==========================================================
# train_models/train_phobert_final.py
# Fine-tuning PhoBERT (vinai/phobert-base-v2) trên CPU – Tối ưu cho máy yếu (Windows)
# ==========================================================

import os, json, numpy as np, pandas as pd, torch, shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import matplotlib.pyplot as plt

# --- Patch lỗi keep_torch_compile trên Windows/CPU ---
import accelerate
if not hasattr(accelerate.Accelerator, "_patched_unwrap"):
    def patched_unwrap_model(self, model, *args, **kwargs):
        # Bỏ qua tham số keep_torch_compile nếu có (fix lỗi trên transformers>=4.45)
        return model
    accelerate.Accelerator.unwrap_model = patched_unwrap_model
    accelerate.Accelerator._patched_unwrap = True
# ------------------------------------------------------

# ===== 0. Thiết lập thư mục cache HuggingFace =====
cache_dir = os.getenv("HF_HOME", "E:\\huggingface_cache")
os.makedirs(cache_dir, exist_ok=True)

# Kiểm tra dung lượng trống
total, used, free = shutil.disk_usage(cache_dir)
if free < 1 * 1024 * 1024 * 1024:  # <1GB
    print(f"⚠️ Cảnh báo: dung lượng trống tại {cache_dir} chỉ còn {free / (1024**2):.1f} MB. Vui lòng giải phóng ổ đĩa!")
else:
    print(f"💾 Dung lượng trống tại {cache_dir}: {free / (1024**3):.2f} GB")

# ===== 1. Đường dẫn & thư mục =====
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "data_balanced.csv")
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# ===== 2. Đọc dữ liệu =====
df = pd.read_csv(DATA).dropna(subset=["content", "label"])
label2id = {lbl: i for i, lbl in enumerate(sorted(df["label"].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# ✅ Dùng 30% dữ liệu để train nhanh hơn
df = df.sample(frac=0.3, random_state=42)

# ===== 3. Chia train/val/test =====
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

train_ds = Dataset.from_pandas(train_df[["content", "label_id"]])
val_ds   = Dataset.from_pandas(val_df[["content", "label_id"]])
test_ds  = Dataset.from_pandas(test_df[["content", "label_id"]])

train_ds = train_ds.rename_columns({"content": "text", "label_id": "labels"})
val_ds   = val_ds.rename_columns({"content": "text", "label_id": "labels"})
test_ds  = test_ds.rename_columns({"content": "text", "label_id": "labels"})

# ===== 4. Tokenizer PhoBERT-base-v2 =====
model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache_dir)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds   = val_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

# ===== 5. Khởi tạo model PhoBERT =====
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    cache_dir=cache_dir
)

# ⚙️ Dùng CPU hoàn toàn
device = torch.device("cpu")
model.to(device)
print(f"⚙️ Huấn luyện trên thiết bị: {device}")

# ===== 6. Tham số huấn luyện =====
args = TrainingArguments(
    output_dir=os.path.join(MODELS, "phobert_base_v2_cpu"),
    eval_strategy="no",               # transformers >=4.47
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir=os.path.join(BASE, "logs"),
    report_to=[],
    dataloader_num_workers=0,
    resume_from_checkpoint=False,
    disable_tqdm=False
)

# ===== 7. Hàm tính chỉ số =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# ===== 8. Huấn luyện PhoBERT =====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ===== 9. Lưu model & tokenizer =====
save_path = os.path.join(MODELS, "phobert_base_v2_cpu_final")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# ===== 10. Đánh giá trên tập test =====
preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
rep = classification_report(y_true, y_pred, target_names=sorted(label2id.keys()), output_dict=True)

# ===== 11. Lưu confusion matrix =====
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=sorted(label2id.keys())).plot(values_format="d")
plt.title("Confusion Matrix - PhoBERT base v2 (CPU)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "cm_test_PhoBERT_base_v2.png"), dpi=160)
plt.close()

# ===== 12. Lưu summary =====
summary = {
    "model": "PhoBERT base v2 (vinai/phobert-base-v2)",
    "device": str(device),
    "test_accuracy": round(acc, 4),
    "test_macro_f1": round(macro_f1, 4),
    "classification_report": rep,
}
with open(os.path.join(REPORTS, "summary_phobert_base_v2.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"✅ Huấn luyện hoàn tất | Accuracy={acc:.4f}, F1_macro={macro_f1:.4f}")
print(f"📁 Model đã lưu tại: {save_path}")
