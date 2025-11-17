# ==========================================================
# train_models/train_baselines.py
# Huấn luyện & so sánh mô hình baseline (Naive Bayes, Logistic Regression, Linear SVM)
# Dành cho project_plcamxuc — dữ liệu: data/data_balanced.csv
# ĐÃ FIX toàn bộ cảnh báo (solver + multi_class)
# ==========================================================

import os, time, json, random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

# ===== 0) CÀI ĐẶT CƠ BẢN =====
random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "data_balanced.csv")
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# ===== 1) ĐỌC DỮ LIỆU =====
df = pd.read_csv(DATA)
df = df.dropna(subset=["content", "label"])
X = df["content"].astype(str)
y = df["label"].astype(str)

# ===== 2) CHIA TRAIN/VAL/TEST =====
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("📊 TỔNG QUAN DỮ LIỆU SAU TIỀN XỬ LÝ:")
print(f"   Tổng số mẫu: {len(df)}")
print(f"   ├─ Train: {len(X_train)} mẫu ({len(X_train)/len(df)*100:.1f}%)")
print(f"   ├─ Val  : {len(X_val)} mẫu ({len(X_val)/len(df)*100:.1f}%)")
print(f"   └─ Test : {len(X_test)} mẫu ({len(X_test)/len(df)*100:.1f}%)")

def label_dist(y):
    return y.value_counts(normalize=True).mul(100).round(2).to_dict()

print("\n🔹 Phân bố nhãn (%):")
print(f"   Train: {label_dist(y_train)}")
print(f"   Val  : {label_dist(y_val)}")
print(f"   Test : {label_dist(y_test)}")
print("=============================================\n")

# ===== 3) TF-IDF VECTORIZER =====
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

joblib.dump(vectorizer, os.path.join(MODELS, "vectorizer.pkl"))

# ===== 4) KHAI BÁO MÔ HÌNH =====
models = {
    "NaiveBayes": MultinomialNB(),
    # ✅ FIX: solver 'lbfgs', bỏ multi_class để tránh cảnh báo sklearn >=1.5
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),
    "LinearSVM": LinearSVC()
}

results = {}

# ===== 5) HÀM HUẤN LUYỆN & ĐÁNH GIÁ =====
def train_eval_save(name, model):
    print(f"🔹 Training {name} ...")
    t0 = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - t0

    # --- Validation ---
    y_pred_val = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred_val)
    macro_f1 = f1_score(y_val, y_pred_val, average="macro")
    rep_val = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)

    # --- Lưu model ---
    model_path = os.path.join(MODELS, f"{name}.pkl")
    joblib.dump(model, model_path)
    size_kb = os.path.getsize(model_path) / 1024

    # --- Lưu confusion matrix ---
    cm = confusion_matrix(y_val, y_pred_val, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (val) - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, f"cm_val_{name}.png"), dpi=160)
    plt.close()

    # --- Lưu pipeline (vectorizer + model) ---
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model)
    ])
    pipe_path = os.path.join(MODELS, f"{name}_pipeline.pkl")
    joblib.dump(pipeline, pipe_path)

    # --- Tổng hợp kết quả ---
    results[name] = {
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "train_time_sec": round(train_time, 2),
        "model_size_kb": round(size_kb, 1),
        "classification_report_val": rep_val
    }

# ===== 6) TRAIN TOÀN BỘ MÔ HÌNH =====
for name, model in models.items():
    train_eval_save(name, model)

# ===== 7) CHỌN BEST MODEL =====
best_name = max(results, key=lambda k: results[k]["val_macro_f1"])
best_model = joblib.load(os.path.join(MODELS, f"{best_name}.pkl"))
print(f"\n🔥 Best model (val macro-F1): {best_name}")

# Nếu best là SVM → calibrate để có xác suất
calibrated_path = None
if best_name == "LinearSVM":
    print("⏳ Calibrating LinearSVM for probability output...")
    cal = CalibratedClassifierCV(base_estimator=LinearSVC(), method="sigmoid", cv=5)
    cal.fit(X_train_vec, y_train)
    calibrated_path = os.path.join(MODELS, "svm_calibrated.pkl")
    joblib.dump(cal, calibrated_path)

# ===== 8) ĐÁNH GIÁ TEST =====
y_pred_test = best_model.predict(X_test_vec)
test_acc = accuracy_score(y_test, y_pred_test)
test_macro_f1 = f1_score(y_test, y_pred_test, average="macro")
rep_test = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

# --- Lưu confusion matrix test ---
cm_test = confusion_matrix(y_test, y_pred_test, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=sorted(y.unique()))
disp.plot(values_format="d")
plt.title(f"Confusion Matrix (test) - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, f"cm_test_{best_name}.png"), dpi=160)
plt.close()

# ===== 9) LƯU SUMMARY =====
summary = {
    "val_results": results,
    "best_model": best_name,
    "test_accuracy": round(test_acc, 4),
    "test_macro_f1": round(test_macro_f1, 4),
    "test_classification_report": rep_test,
    "calibrated_svm_path": calibrated_path
}

with open(os.path.join(REPORTS, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ===== 10) IN KẾT QUẢ =====
print("\n=== 📋 VALIDATION RESULTS ===")
res_df = pd.DataFrame([
    {
        "Model": n,
        "Accuracy": r["val_accuracy"],
        "Macro_F1": r["val_macro_f1"],
        "Train_Time(s)": r["train_time_sec"],
        "Size(KB)": r["model_size_kb"]
    } for n, r in results.items()
]).sort_values(by="Macro_F1", ascending=False)
print(res_df.to_string(index=False))

print(f"\n🔥 BEST MODEL: {best_name}")
print(f"📈 Test Accuracy: {test_acc:.4f} | Test Macro-F1: {test_macro_f1:.4f}")
print("📁 Models  :", MODELS)
print("📁 Reports :", REPORTS)
print("\n✅ Training pipeline completed successfully.")

# ===== 11) BIỂU ĐỒ SO SÁNH CÁC MÔ HÌNH =====
metrics_df = pd.DataFrame([
    {"Model": n, "Accuracy": r["val_accuracy"], "Macro_F1": r["val_macro_f1"]}
    for n, r in results.items()
]).melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(7, 5))
for metric in ["Accuracy", "Macro_F1"]:
    subset = metrics_df[metrics_df["Metric"] == metric]
    plt.bar(subset["Model"], subset["Score"], alpha=0.7, label=metric)

plt.title("So sánh hiệu suất các mô hình baseline")
plt.ylabel("Giá trị (%)")
plt.ylim(0.8, 0.9)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, "baseline_comparison.png"), dpi=160)
plt.close()

print("📊 Saved performance comparison chart → baseline_comparison.png")
print("📊 Saved test confusion matrix → cm_test_{best_name}.png")
