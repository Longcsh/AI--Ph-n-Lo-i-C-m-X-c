# ==========================================================
# app_phobert.py
# Giao diện dự đoán cảm xúc tiếng Việt bằng mô hình PhoBERT
# ==========================================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ===== 1️⃣ Cấu hình ban đầu =====
st.set_page_config(page_title="Phân tích cảm xúc tiếng Việt", page_icon="🧠", layout="centered")

st.title("🧠 Phân tích cảm xúc tiếng Việt với PhoBERT-base-v2")
st.markdown(
    """
    Ứng dụng này sử dụng **PhoBERT-base-v2** (fine-tuned trên tập bình luận tiếng Việt)  
    để phân loại cảm xúc của văn bản thành **Tích cực**, **Tiêu cực**, hoặc **Trung lập**.
    """
)

# ===== 2️⃣ Tải mô hình PhoBERT =====
@st.cache_resource
def load_model():
    model_dir = os.path.join("models", "phobert_base_v2_cpu_final")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label  # ví dụ: {0: 'negative', 1: 'neutral', 2: 'positive'}

# ===== 3️⃣ Hàm dự đoán =====
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
    return id2label[pred_id], probs

# ===== 4️⃣ Giao diện nhập văn bản =====
st.subheader("✍️ Nhập đoạn văn cần phân tích:")
user_input = st.text_area("Ví dụ: Sản phẩm này dùng rất tốt, mình sẽ ủng hộ lần sau!", height=120)

if st.button("🚀 Phân tích cảm xúc"):
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập nội dung trước khi phân tích.")
    else:
        label, probs = predict_sentiment(user_input)

        # 🌈 Tô màu cảm xúc
        color_map = {
            "positive": "#22c55e",
            "neutral": "#facc15",
            "negative": "#ef4444"
        }
        st.success(f"**Kết quả dự đoán:** 🧩 {label.capitalize()}")
        df = pd.DataFrame({
            "Cảm xúc": [id2label[i] for i in range(len(probs))],
            "Xác suất (%)": np.round(probs * 100, 2)
        })

        # ===== Biểu đồ cột đẹp bằng Plotly =====
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["Cảm xúc"],
                    y=df["Xác suất (%)"],
                    marker_color=[color_map[df["Cảm xúc"][i]] for i in range(len(df))],
                    text=df["Xác suất (%)"],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Xác suất từng nhãn cảm xúc",
            xaxis_title="Nhãn cảm xúc",
            yaxis_title="Xác suất (%)",
            template="simple_white"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("📘 Mô hình: vinai/phobert-base-v2 | Tác giả: nhóm dự án phân tích cảm xúc tiếng Việt (2025)")
