# 🧠 BERT-Based Sentiment Analysis

<p align="center">
  <b>End-to-End NLP Pipeline for Binary Sentiment Classification</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Deployment-red?logo=streamlit">
</p>

---

## 📌 Overview

This project implements an end-to-end **Natural Language Processing (NLP) pipeline** for binary sentiment classification, comparing traditional machine learning approaches with modern transformer-based architectures.

The project evaluates three different modeling approaches:

- 📊 **TF-IDF + Logistic Regression**
- 🧊 **Frozen BERT**
- 🔥 **Fully Fine-Tuned BERT**

The complete workflow includes data validation, preprocessing, reproducible training, hyperparameter experimentation, early stopping, checkpoint restoration, model comparison, error analysis, and interactive deployment using Streamlit.

---

## 🎯 Problem Statement

The objective is to classify an English sentence into one of two sentiment categories:

| Label | Sentiment |
|:---:|:---|
| `0` | ❌ Negative |
| `1` | ✅ Positive |

The project investigates how contextual representations learned by BERT compare with traditional TF-IDF-based text representations for sentiment classification.

---

## 🚀 Key Features

- 🧹 Automated dataset validation and preprocessing
- 📊 TF-IDF + Logistic Regression baseline
- 🧊 Frozen BERT feature-extraction model
- 🔥 Fully Fine-Tuned BERT classifier
- 🔍 Learning-rate experimentation
- 🔢 Epoch-count experimentation
- ⏹️ Validation-based early stopping
- 💾 Best-model checkpoint restoration
- 🎲 Reproducible training with fixed random seeds
- 📏 Dynamic BERT sequence-length selection
- ⚡ CUDA mixed-precision training support
- 📈 Comprehensive model evaluation
- 🔎 Per-example error analysis
- 🌐 Interactive Streamlit inference application

---

# 🏗️ Model Architecture

## 1️⃣ TF-IDF + Logistic Regression

A classical NLP pipeline is implemented as the baseline model.

```text
                Input Sentence
                       │
                       ▼
              ┌─────────────────┐
              │ TF-IDF Vectorizer│
              │     1–2 Grams    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Logistic        │
              │ Regression      │
              └────────┬────────┘
                       │
                       ▼
              Positive / Negative
🔹 Purpose

The baseline provides an efficient and interpretable benchmark for evaluating whether transformer-based contextual representations improve sentiment classification performance.

# 2️⃣ Frozen BERT
In this approach, the pretrained BERT encoder is kept frozen while only the classification head is trained.
```text
                 Input Text
                     │
                     ▼
              ┌───────────────┐
              │ BERT Tokenizer│
              └──────┬────────┘
                     │
                     ▼
              ┌───────────────┐
              │ Pretrained    │
              │ BERT Encoder  │
              │    FROZEN     │
              └──────┬────────┘
                     │
                     ▼
              ┌───────────────┐
              │ CLS Embedding │
              └──────┬────────┘
                     │
                     ▼
              ┌───────────────┐
              │ Dropout       │
              └──────┬────────┘
                     │
                     ▼
              ┌───────────────┐
              │ Linear Head   │
              └──────┬────────┘
                     │
                     ▼
              Positive / Negative

🔹 Purpose

This experiment measures the effectiveness of pretrained BERT representations without updating the transformer encoder parameters.
