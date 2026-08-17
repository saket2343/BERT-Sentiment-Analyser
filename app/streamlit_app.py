"""
Streamlit application for the BERT Sentiment Analyzer.

Loads the saved best fine-tuned BERT model (and, if available, the TF-IDF
baseline and Frozen BERT model for comparison). Never trains anything.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st
import torch

# Allow running `streamlit run app/streamlit_app.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BEST_BERT_DIR, FROZEN_BERT_DIR, TFIDF_MODEL_DIR, LABEL_MAP, DEVICE
from src.bert_model import load_model
from src.baseline import load_pipeline


st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="🎯", layout="centered")


@st.cache_resource(show_spinner=False)
def get_bert_model(model_dir: str):
    try:
        model, tokenizer, config = load_model(model_dir, device=DEVICE)
        return model, tokenizer, config
    except FileNotFoundError:
        return None, None, None


@st.cache_resource(show_spinner=False)
def get_tfidf_pipeline():
    try:
        return load_pipeline(TFIDF_MODEL_DIR)
    except FileNotFoundError:
        return None


@torch.no_grad()
def predict_with_bert(model, tokenizer, config, text: str):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=config["max_length"],
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    model.eval()
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return LABEL_MAP[pred_idx], confidence


def predict_with_tfidf(pipeline, text: str):
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    confidence = float(proba[int(pred)])
    return LABEL_MAP[int(pred)], confidence


def main():
    st.title("🎯 BERT Sentiment Analyzer")
    st.caption("Binary sentiment classification with a fine-tuned BERT model.")

    best_model, best_tokenizer, best_config = get_bert_model(BEST_BERT_DIR)

    if best_model is None:
        st.error(
            f"No trained model found at `{BEST_BERT_DIR}`. "
            f"Train the final BERT model first — see README.md for the exact command."
        )
        st.stop()

    text = st.text_area(
        "Enter an English sentence:",
        placeholder="e.g. This movie completely blew me away, I loved every minute of it.",
        height=120,
    )

    show_comparison = st.checkbox("Compare with other models", value=False)

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            label, confidence = predict_with_bert(best_model, best_tokenizer, best_config, text)

            st.subheader("Prediction")
            color = "green" if label == "Positive" else "red"
            st.markdown(f"### :{color}[{label.upper()}]")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.caption("Model: Fine-Tuned BERT")

            if show_comparison:
                st.divider()
                st.subheader("Model Comparison")

                tfidf_pipeline = get_tfidf_pipeline()
                frozen_model, frozen_tokenizer, frozen_config = get_bert_model(FROZEN_BERT_DIR)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**TF-IDF + Logistic Regression**")
                    if tfidf_pipeline is not None:
                        t_label, t_conf = predict_with_tfidf(tfidf_pipeline, text)
                        st.write(f"Prediction: **{t_label}**")
                        st.write(f"Confidence: {t_conf * 100:.1f}%")
                    else:
                        st.write("Not trained yet.")

                with col2:
                    st.markdown("**Frozen BERT**")
                    if frozen_model is not None:
                        f_label, f_conf = predict_with_bert(
                            frozen_model, frozen_tokenizer, frozen_config, text
                        )
                        st.write(f"Prediction: **{f_label}**")
                        st.write(f"Confidence: {f_conf * 100:.1f}%")
                    else:
                        st.write("Not trained yet.")

                with col3:
                    st.markdown("**Fully Fine-Tuned BERT**")
                    st.write(f"Prediction: **{label}**")
                    st.write(f"Confidence: {confidence * 100:.1f}%")

    with st.expander("About this app"):
        st.write(
            "This app loads a pre-trained, saved BERT sentiment classifier. "
            "It does not train or fine-tune any model at runtime. See the "
            "project README for how the model was trained and evaluated."
        )


if __name__ == "__main__":
    main()
