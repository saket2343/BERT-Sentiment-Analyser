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

This project implements an **end-to-end sentiment classification pipeline** that compares traditional machine learning with modern transformer-based NLP models.

The project systematically evaluates three approaches:

- 📊 **TF-IDF + Logistic Regression**
- 🧊 **Frozen BERT**
- 🔥 **Fully Fine-Tuned BERT**

The pipeline goes beyond basic model training by incorporating **data validation, reproducible experiments, hyperparameter search, early stopping, checkpoint restoration, model comparison, error analysis, and Streamlit deployment**.

---

## 🎯 Problem Statement

The objective is to classify an English sentence into one of two sentiment categories:

| Label | Sentiment |
|:---:|:---|
| `0` | ❌ Negative |
| `1` | ✅ Positive |

The project investigates how contextual representations from BERT compare against traditional TF-IDF-based representations for sentiment classification.

---

## 🚀 Key Features

- 🧹 Automated dataset validation and preprocessing
- 📊 TF-IDF + Logistic Regression baseline
- 🧊 Frozen BERT feature-extraction approach
- 🔥 Fully fine-tuned BERT classifier
- 🔍 Learning-rate experimentation
- 🔢 Epoch-count experimentation
- ⏹️ Validation-based early stopping
- 💾 Best-model checkpoint restoration
- 🎲 Reproducible training with fixed random seeds
- 📏 Dynamic BERT sequence-length selection
- ⚡ CUDA mixed-precision support
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
              │    1–2 Grams     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Logistic         │
              │ Regression       │
              └────────┬────────┘
                       │
                       ▼
              Positive / Negative
