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

# 🧠 BERT-Based Sentiment Analysis

## 📌 Overview

This project implements an end-to-end **Natural Language Processing (NLP) pipeline** for binary sentiment classification, comparing traditional machine learning approaches with modern transformer-based architectures.

The project evaluates three different modeling approaches:

- 📊 **TF-IDF + Logistic Regression**
- 🧊 **Frozen BERT**
- 🔥 **Fully Fine-Tuned BERT**

The complete workflow includes **data validation, preprocessing, reproducible training, hyperparameter experimentation, early stopping, checkpoint restoration, model comparison, error analysis, and interactive deployment using Streamlit**.

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
                 ┌──────────────────┐
                 │ TF-IDF Vectorizer│
                 │     1–2 Grams    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Logistic         │
                 │ Regression       │
                 └────────┬─────────┘
                          │
                          ▼
                 Positive / Negative
```

### 🔹 Purpose

The TF-IDF baseline provides a fast and interpretable benchmark for evaluating the effectiveness of transformer-based models.

---

## 2️⃣ Frozen BERT

In this approach, the pretrained BERT encoder is used as a **fixed feature extractor**, while only the classification head is trained.

```text
                    Input Text
                        │
                        ▼
                ┌──────────────┐
                │ BERT Tokenizer│
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Pretrained   │
                │ BERT Encoder │
                │    FROZEN    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ CLS Embedding│
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   Dropout    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Linear Head  │
                └──────┬───────┘
                       │
                       ▼
                Positive / Negative
```

### 🔹 Purpose

This experiment measures how effectively pretrained BERT representations perform without updating the transformer encoder during task-specific training.

---

## 3️⃣ Fully Fine-Tuned BERT

The complete BERT encoder and classification head are jointly optimized for the sentiment classification task.

```text
                    Input Text
                        │
                        ▼
                ┌──────────────┐
                │ BERT Tokenizer│
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ BERT Encoder │
                │  FINE-TUNED  │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ CLS Embedding│
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   Dropout    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Linear Head  │
                └──────┬───────┘
                       │
                       ▼
                Positive / Negative
```

### 🔥 Training Components

- ⚙️ AdamW optimizer
- 📉 Learning-rate scheduling
- ✂️ Gradient clipping
- ⏹️ Early stopping
- 💾 Best-checkpoint restoration
- 📦 Configurable batch sizes
- 🔄 Gradient accumulation
- ⚡ CUDA mixed-precision training

---

# 📊 Experimental Design

The project follows a controlled experimentation workflow to compare different model configurations and training hyperparameters.

---

## 🔍 Learning-Rate Search

The fine-tuned BERT model can be evaluated across multiple learning rates:

```text
1e-5
2e-5
3e-5
5e-5
```

Results are stored in:

```text
experiments/learning_rate_results.csv
```

---

## 🔢 Epoch Search

Different training durations can be evaluated using:

```text
2 Epochs
3 Epochs
4 Epochs
5 Epochs
```

Results are stored in:

```text
experiments/epoch_results.csv
```

---

## ⏹️ Early Stopping

Training monitors the **validation F1 score** rather than the test set.

The checkpoint with the best validation F1 score is restored after training.

### Benefits

- 🛡️ Reduces overfitting
- ⏱️ Avoids unnecessary training
- 🎯 Selects the strongest validation checkpoint
- 🔒 Keeps the test set isolated during model selection

---

# 📈 Model Evaluation

All models are evaluated on the same untouched test split.

## 📌 Evaluation Metrics

| Metric | Description |
|:---|:---|
| **Accuracy** | Overall classification correctness |
| **Precision** | Reliability of positive predictions |
| **Recall** | Coverage of actual positive samples |
| **F1 Score** | Harmonic mean of precision and recall |
| **Macro F1** | Average F1 score across both classes |

---

## 🤖 Models Compared

```text
                     Test Dataset
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │   TF-IDF   │ │   Frozen   │ │ Fine-Tuned │
     │ + Logistic │ │    BERT    │ │    BERT    │
     │ Regression │ │            │ │            │
     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Model Comparison│
                 └─────────────────┘
```

Results are stored in:

```text
experiments/model_comparison.csv
```

---

# 🔎 Error Analysis

The project performs **per-example error analysis** to understand where the model succeeds and fails rather than relying only on aggregate evaluation metrics.

For every test example, the following information is recorded:

| Field | Description |
|:---|:---|
| `sentence` | Input text |
| `actual_label` | Ground-truth sentiment |
| `predicted_label` | Model prediction |
| `confidence` | Prediction confidence |
| `correct` | Whether the prediction was correct |

### 🔍 Error Patterns

The analysis helps identify challenging examples involving:

- ❌ Negation
- 🔀 Mixed or contrasting sentiment
- 📏 Long text sequences
- ⚠️ High-confidence incorrect predictions

Output file:

```text
results/error_analysis.csv
```

---

# 🎲 Reproducibility

The project uses a fixed random seed:

```text
Seed = 42
```

The seed is applied to:

- 🐍 Python random operations
- 🔢 NumPy
- 🔥 PyTorch CPU operations
- 🎮 PyTorch CUDA operations
- 📊 Dataset splitting

This ensures consistent dataset splits and improves reproducibility across experiments.

---

# ⚡ Compute & Memory Optimization

The training pipeline is designed to adapt to the available hardware.

## 🖥️ Hardware-Aware Training

```text
                    ┌───────────────┐
                    │   Hardware    │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          ┌────────────┐        ┌────────────┐
          │  CUDA GPU  │        │    CPU     │
          └─────┬──────┘        └─────┬──────┘
                │                     │
                ▼                     ▼
        Mixed Precision         CPU Training
```

## ⚙️ Optimization Techniques

- ⚡ CUDA mixed precision / FP16
- 📦 Gradient accumulation
- 🔢 Configurable batch sizes
- 📊 Configurable evaluation batch sizes
- ✂️ Gradient clipping
- 📏 Dynamic padding
- 🚀 Dynamic sequence-length selection

---

# 📏 Dynamic Sequence Length

Instead of automatically using BERT's maximum sequence length of `512`, the pipeline analyzes the token-length distribution of the training dataset.

An appropriate sequence length is selected based on the observed data distribution.

### 🎯 Benefits

- ⚡ Faster training
- 💾 Lower memory consumption
- 📉 Less unnecessary padding
- 🚀 More efficient inference

---

# 📂 Project Structure

```text
BERT-Sentiment-Analyser/
│
├── 📁 app/
│   └── streamlit_app.py
│
├── 📁 data/
│   └── sentiment_train.csv
│
├── 📁 experiments/
│   ├── learning_rate_results.csv
│   ├── epoch_results.csv
│   ├── model_comparison.csv
│   └── README.md
│
├── 📁 models/
│   ├── tfidf_logistic/
│   ├── frozen_bert/
│   └── best_bert/
│
├── 📁 notebooks/
│   └── sentiment_analysis_experiments.ipynb
│
├── 📁 results/
│   └── error_analysis.csv
│
├── 📁 src/
│   ├── __init__.py
│   ├── config.py
│   ├── seed.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── bert_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── error_analysis.py
│
├── 📄 .gitignore
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|:---|:---|
| 🐍 **Programming** | Python 3.9+ |
| 🤖 **Deep Learning** | PyTorch |
| 🧠 **NLP** | BERT, Hugging Face Transformers |
| 📊 **Machine Learning** | Scikit-learn |
| 🧹 **Data Processing** | Pandas, NumPy |
| 📈 **Visualization** | Matplotlib, Seaborn |
| 🌐 **Deployment** | Streamlit |
| 📓 **Experimentation** | Jupyter Notebook |

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd BERT-Sentiment-Analyser
```

---

## 2️⃣ Create a Virtual Environment

### 🍎 macOS / 🐧 Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 🪟 Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🗃️ Dataset Setup

Place the dataset in:

```text
data/
└── sentiment_train.csv
```

The CSV file should contain the following columns:

| Column | Description |
|:---|:---|
| `sentence` | Input English sentence |
| `label` | Binary sentiment label |

Expected labels:

```text
0 → Negative
1 → Positive
```

The data pipeline automatically validates:

- 📄 File existence
- 🏷️ Required columns
- ❓ Missing values
- ♻️ Duplicate records
- ⚖️ Label distribution
- 📝 Text length
- 🔤 Token length

---

# 🚀 Training Pipeline

## 1️⃣ Validate the Dataset

```bash
python -m src.data
```

This performs dataset validation and reports important dataset statistics.

---

## 2️⃣ Train TF-IDF Baseline

```bash
python -m src.baseline
```

The trained baseline model is saved under:

```text
models/tfidf_logistic/
```

---

## 3️⃣ Train Frozen BERT

```bash
python -m src.train \
    --freeze_bert \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/frozen_bert
```

---

## 4️⃣ Run Learning-Rate Search

```bash
python -m src.train \
    --mode lr_search \
    --epochs 3
```

Results:

```text
experiments/learning_rate_results.csv
```

---

## 5️⃣ Run Epoch Search

After selecting the preferred learning rate:

```bash
python -m src.train \
    --mode epoch_search \
    --learning_rate 2e-5
```

Results:

```text
experiments/epoch_results.csv
```

---

## 6️⃣ Train Fully Fine-Tuned BERT

```bash
python -m src.train \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/best_bert
```

---

# 📊 Model Evaluation

## 🔬 Compare All Models

```bash
python -m src.evaluate --compare_all
```

The evaluation compares:

- 📊 TF-IDF + Logistic Regression
- 🧊 Frozen BERT
- 🔥 Fully Fine-Tuned BERT

Results are saved to:

```text
experiments/model_comparison.csv
```

---

## 🧪 Evaluate the Final BERT Model

```bash
python -m src.evaluate \
    --model_dir models/best_bert
```

---

# 🔍 Run Error Analysis

```bash
python -m src.error_analysis \
    --model_dir models/best_bert
```

Output:

```text
results/error_analysis.csv
```

---

# 🌐 Streamlit Application

The project includes an interactive web application for real-time sentiment prediction.

## ▶️ Launch the Application

```bash
streamlit run app/streamlit_app.py
```

The application allows users to:

1. ✍️ Enter an English sentence
2. 🧠 Generate a sentiment prediction
3. 📊 View prediction confidence
4. 🔄 Compare available models

### 💬 Example

```text
Input:
"This movie was absolutely fantastic."

Prediction:
✅ POSITIVE
```

> **Note:** Prediction confidence depends on the trained model and input text.

---

# 🏭 Deployment Architecture

```text
                    ┌──────────────────────┐
                    │      User Input      │
                    │    English Text      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Interface  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    BERT Tokenizer    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Fine-Tuned BERT    │
                    │       Model          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sentiment Prediction │
                    │ + Confidence Score   │
                    └──────────────────────┘
```

---

# 💡 Key Engineering Decisions

## 1️⃣ Baseline Before Deep Learning

A **TF-IDF + Logistic Regression** model establishes a simple and interpretable benchmark before introducing transformer architectures.

---

## 2️⃣ Frozen vs Fine-Tuned BERT

Both configurations are evaluated to distinguish the value of pretrained representations from the additional benefit of task-specific fine-tuning.

---

## 3️⃣ Validation-Based Model Selection

Hyperparameters are selected using validation performance while the test set remains isolated for final evaluation.

---

## 4️⃣ F1-Based Early Stopping

Validation F1 is used for checkpoint selection to provide a balanced measure of classification performance.

---

## 5️⃣ Reproducible Experiments

Fixed seeds and consistent data splitting improve experiment reproducibility and make model comparisons more reliable.

---

## 6️⃣ Production-Oriented Inference

The Streamlit application loads trained model checkpoints and performs inference without retraining the models during application startup.

---

# 📈 Results

Model performance is generated automatically during evaluation.

| Model | Accuracy | Precision | Recall | F1 Score | Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 📊 TF-IDF + Logistic Regression | — | — | — | — | — |
| 🧊 Frozen BERT | — | — | — | — | — |
| 🔥 Fine-Tuned BERT | — | — | — | — | — |

> 📌 **Note:** Run the training and evaluation pipeline to populate the table with the actual experiment results.

---

# 🔮 Future Improvements

- 🤏 **DistilBERT** for faster inference
- 🚀 **ONNX / TorchScript** model optimization
- 🌐 **FastAPI** inference service
- 🐳 **Dockerized** deployment
- 📊 **MLflow / Weights & Biases** experiment tracking
- ⚖️ Class-weighted training
- 🎯 Prediction probability calibration
- 📦 Batch inference support
- 🔄 Automated CI/CD testing
- 🧪 Robustness and adversarial testing

---

# 📜 License

This project is intended for **educational, research, and portfolio purposes**.

If distributing the project publicly, consider adding an appropriate open-source license such as the **MIT License**.

---

# 👨‍💻 Author

## Saket Pandey

🎓 **IIT Madras**  
🧬 **Biotechnology / Computational Biology**

---

<p align="center">
  ⭐ If you found this project useful, consider giving the repository a star!
</p>

<p align="center">
  <b>Built with Python • PyTorch • Hugging Face • Scikit-learn • Streamlit</b>
</p>
