🧠 BERT-Based Sentiment Analysis

Production-Oriented NLP Classification Pipeline with Baseline Benchmarking, Transformer Fine-Tuning & Deployment



📖 Overview

This project implements an end-to-end sentiment classification system for binary sentiment prediction using classical machine learning and transformer-based NLP models.

The pipeline benchmarks three progressively stronger approaches:

TF-IDF + Logistic Regression — classical NLP baseline

Frozen BERT — pretrained BERT used as a fixed feature extractor

Fine-Tuned BERT — end-to-end task-specific transformer fine-tuning

Beyond model training, the project includes data validation, stratified splitting, token-length analysis, hyperparameter experimentation, early stopping, checkpoint management, final evaluation, error analysis, and Streamlit-based inference.

The objective is to establish a reproducible workflow for measuring the practical improvement provided by pretrained transformer representations and task-specific fine-tuning over a strong classical baseline.

✨ Features

🧹 Data Processing

Automated dataset validation and profiling

Missing-value and duplicate detection

Label-distribution analysis

Stratified train/validation/test splitting

Automatic text-length and token-length analysis

🤖 Model Development

TF-IDF + Logistic Regression baseline

Frozen BERT feature-extraction pipeline

Fully fine-tuned bert-base-uncased

Configurable learning rate, batch size and epochs

Early stopping with best-checkpoint restoration

Gradient clipping

Optional mixed-precision GPU training

Gradient accumulation for memory-constrained environments

🔬 Experimentation

Learning-rate comparison

Epoch comparison

Validation-based model selection

Consistent evaluation across all model variants

Reproducible experiments using fixed random seeds

📊 Evaluation & Analysis

Accuracy

Precision

Recall

F1-score

Macro F1-score

Confusion matrix

Prediction confidence

Per-example error analysis

Comparison of classical and transformer-based approaches

🚀 Deployment

Saved model checkpoints

Reusable tokenizer and configuration

Lightweight Streamlit inference application

CPU/GPU-aware inference

🏗️ System Architecture

                         ┌──────────────────────┐
                         │     Input Dataset    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Validation &    │
                         │ Dataset Profiling    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Stratified Split     │
                         │ 70% / 15% / 15%      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
              ┌──────────┐   ┌────────────┐   ┌──────────────┐
              │  TF-IDF  │   │ Frozen     │   │ Fine-Tuned   │
              │ + LR     │   │ BERT       │   │ BERT         │
              └────┬─────┘   └─────┬──────┘   └──────┬───────┘
                   │               │                 │
                   └───────────────┼─────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Validation-Based     │
                         │ Model Selection      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Final Test           │
                         │ Evaluation            │
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                ┌──────────────┐         ┌────────────────┐
                │ Error        │         │ Saved Model    │
                │ Analysis     │         │ Artifacts      │
                └──────────────┘         └───────┬────────┘
                                                 │
                                                 ▼
                                         ┌────────────────┐
                                         │ Streamlit      │
                                         │ Inference App  │
                                         └────────────────┘

🧠 Modeling Approach

1. TF-IDF + Logistic Regression

A classical NLP baseline is established using TF-IDF representations with unigram and bigram features.

Input Text
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Sentiment Prediction

This provides a transparent benchmark against which transformer-based approaches can be evaluated.

2. Frozen BERT

The second approach uses pretrained bert-base-uncased as a fixed feature extractor.

Input Text
    ↓
BERT Tokenizer
    ↓
Pretrained BERT Encoder
    ↓
[CLS] Representation
    ↓
Classification Head
    ↓
Sentiment

The BERT encoder remains frozen while the classification head is trained for the sentiment task.

3. Fully Fine-Tuned BERT

The final approach fine-tunes the complete BERT encoder together with the classification head.

Input Text
    ↓
BERT Tokenizer
    ↓
BERT Encoder
    ↓
[CLS] Representation
    ↓
Dropout
    ↓
Classification Head
    ↓
Positive / Negative

Training supports:

AdamW optimization

Learning-rate scheduling

Warmup

Gradient clipping

Early stopping

Best validation-checkpoint restoration

Mixed-precision training on CUDA-enabled hardware

Gradient accumulation

The primary model-selection criterion is validation F1, while the test set is reserved for final evaluation.

📂 Dataset

The pipeline expects a CSV file containing:

Column

Description

sentence

Input English sentence

label

Binary sentiment label

Expected label mapping:

0 → Negative
1 → Positive

Place the dataset at:

data/sentiment_train.csv

Note: The dataset itself is not included in the repository.

Data Validation

Before training, the pipeline checks:

Required columns

Missing values

Duplicate sentences

Label distribution

Number of unique labels

Text-length statistics

Token-length statistics

🔀 Data Splitting

The dataset is divided using a stratified:

70% → Training
15% → Validation
15% → Test

A fixed random seed (42) is used to make the split reproducible.

The test set is not used for:

Hyperparameter selection

Epoch selection

Early stopping

Model selection

This maintains a clean final evaluation protocol.

🔤 Tokenization & Sequence Optimization

BERT tokenization is performed using BertTokenizerFast.

Instead of blindly padding every sample to the model's maximum sequence length, the pipeline:

Computes token-length statistics

Estimates the 95th percentile

Selects an appropriate maximum sequence length

Applies an upper bound

Uses dynamic batch-level padding

This reduces unnecessary padding and improves training efficiency.

🔬 Experimentation

Learning-Rate Search

Supported learning rates include:

1e-5
2e-5
3e-5
5e-5

Results are stored in:

experiments/learning_rate_results.csv

Epoch Search

The pipeline supports controlled evaluation across different epoch counts, for example:

2
3
4
5

Results are stored in:

experiments/epoch_results.csv

Model Comparison

The same held-out test set is used for final comparison:

TF-IDF + Logistic Regression
Frozen BERT
Fully Fine-Tuned BERT

Comparison results are stored in:

experiments/model_comparison.csv

📊 Evaluation

The evaluation pipeline reports:

Accuracy

Precision

Recall

F1-score

Macro F1-score

Per-class metrics

Confusion matrix

Prediction confidence

Generated artifacts include:

results/
├── confusion_matrix.png
├── model_comparison.png
├── training_curves.png
└── error_analysis.csv

Results

Metrics should be populated from the actual experiment outputs rather than hardcoded values.

Model

Accuracy

Precision

Recall

F1

Macro F1

TF-IDF + Logistic Regression

—

—

—

—

—

Frozen BERT

—

—

—

—

—

Fully Fine-Tuned BERT

—

—

—

—

—

🔎 Error Analysis

The project includes a dedicated error-analysis pipeline.

For each test example, the analysis records:

sentence
actual_label
predicted_label
confidence
correct

It can be used to inspect:

False positives

False negatives

High-confidence incorrect predictions

Negation-related errors

Contrast or mixed-sentiment language

Long-form inputs

The analysis is intended as a diagnostic tool for identifying recurring model failure patterns.

⚙️ Reproducibility

The project uses a centralized random seed:

Seed = 42

The seed is applied to:

Python random

NumPy

PyTorch CPU

PyTorch CUDA

Data splitting and experiment configuration are also deterministic where supported.

🚀 Training Efficiency

The pipeline automatically detects available hardware:

CUDA GPU → GPU training
No CUDA  → CPU training

Efficiency mechanisms include:

Dynamic padding

Mixed precision on CUDA

Gradient accumulation

Configurable batch sizes

Gradient clipping

Data-driven sequence length

Early stopping

Example:

python -m src.train \
    --epochs 4 \
    --learning_rate 2e-5 \
    --batch_size 16 \
    --eval_batch_size 32

🖥️ Streamlit Deployment

The repository includes a lightweight Streamlit inference interface.

The application loads the saved fine-tuned BERT model and performs inference without retraining.

Launch

streamlit run app/streamlit_app.py

Inference Flow

User Input
    ↓
BERT Tokenizer
    ↓
Saved Fine-Tuned BERT
    ↓
Softmax Probabilities
    ↓
Sentiment + Confidence

📁 Repository Structure

bert-sentiment-analysis/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── sentiment_train.csv
│
├── experiments/
│   ├── learning_rate_results.csv
│   ├── epoch_results.csv
│   ├── model_comparison.csv
│   └── README.md
│
├── models/
│   ├── tfidf_logistic/
│   ├── frozen_bert/
│   └── best_bert/
│
├── notebooks/
│   └── sentiment_analysis_experiments.ipynb
│
├── results/
│   └── generated evaluation artifacts
│
├── src/
│   ├── __init__.py
│   ├── baseline.py
│   ├── bert_model.py
│   ├── config.py
│   ├── data.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── seed.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md

🛠️ Tech Stack

Category

Technologies

Language

Python

Deep Learning

PyTorch

NLP

Hugging Face Transformers

Classical ML

Scikit-learn

Data Processing

Pandas, NumPy

Visualization

Matplotlib, Seaborn

Deployment

Streamlit

Model Serialization

PyTorch, Joblib

Hardware

CPU / CUDA GPU

⚡ Installation

1. Clone the repository

git clone <repository-url>
cd bert-sentiment-analysis

2. Create a virtual environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Windows

python -m venv .venv
.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Add the dataset

Place:

sentiment_train.csv

inside:

data/

▶️ Usage

Run the classical baseline

python -m src.baseline

Train Frozen BERT

python -m src.train \
    --freeze_bert \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/frozen_bert

Run learning-rate search

python -m src.train \
    --mode lr_search \
    --epochs 3

Run epoch search

python -m src.train \
    --mode epoch_search \
    --learning_rate 2e-5

Train the final BERT model

python -m src.train \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/best_bert

Evaluate the final model

python -m src.evaluate \
    --model_dir models/best_bert

Compare all models

python -m src.evaluate --compare_all

Run error analysis

python -m src.error_analysis \
    --model_dir models/best_bert

Launch Streamlit

streamlit run app/streamlit_app.py

💾 Model Artifacts

A trained BERT checkpoint contains the model state, tokenizer configuration, and training metadata required for inference.

Typical artifacts include:

models/best_bert/
├── model_state_dict.pt
├── training_config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer.json
└── vocab.txt

The saved configuration captures parameters such as:

Base BERT model

Number of labels

Dropout

Maximum sequence length

Label mapping

Training configuration

Validation performance

🧩 Engineering Principles

Separation of Concerns

Data processing, preprocessing, modeling, training, evaluation, and deployment are implemented as separate modules.

Centralized Configuration

Paths, model parameters, split configuration, and experiment defaults are maintained centrally.

Validation-Driven Model Selection

Hyperparameters are selected using validation performance, while the test set remains reserved for final evaluation.

Reproducibility

Random seeds, deterministic data splits, and saved configurations support repeatable experiments.

Reusable Inference

Saved model artifacts can be loaded independently of the training workflow.

🔮 Future Improvements

Confidence calibration

Cross-validation for robust model comparison

Automated experiment tracking

Additional transformer architectures

Data augmentation for difficult examples

Threshold optimization

REST API inference

Containerized deployment

CI-based automated testing

Model/version registry integration

📄 License

Add the applicable organization or project license here.

👤 Author

Saket Pandey

Developed as an end-to-end NLP/ML engineering project focused on reproducible experimentation, transformer fine-tuning, model evaluation, and deployment.
