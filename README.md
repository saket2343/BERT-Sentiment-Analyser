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

The TF-IDF baseline provides a fast, interpretable benchmark for evaluating the effectiveness of transformer-based models.

2️⃣ Frozen BERT

In this approach, the pretrained BERT encoder is used as a fixed feature extractor while only the classification head is trained.

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
🔹 Purpose

This experiment measures how effectively pretrained BERT representations perform without updating the transformer encoder during task-specific training.

3️⃣ Fully Fine-Tuned BERT

The complete BERT encoder and classification head are jointly optimized for sentiment classification.

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
🔥 Training Components
AdamW optimizer
Learning-rate scheduling
Gradient clipping
Early stopping
Best-checkpoint restoration
Configurable batch sizes
Gradient accumulation
CUDA mixed-precision training
📊 Experimental Design

The project follows a controlled experimentation workflow to compare different model configurations.

🔍 Learning-Rate Search

The fine-tuned BERT model can be evaluated across multiple learning rates:

1e-5
2e-5
3e-5
5e-5

Results are stored in:

experiments/learning_rate_results.csv
🔢 Epoch Search

Different training durations can be evaluated using:

2 Epochs
3 Epochs
4 Epochs
5 Epochs

Results are stored in:

experiments/epoch_results.csv
⏹️ Early Stopping

Training monitors the validation F1 score rather than the test set.

The checkpoint with the best validation F1 score is restored after training.

This helps to:

🛡️ Reduce overfitting
⏱️ Avoid unnecessary training
🎯 Select the strongest validation checkpoint
📈 Model Evaluation

All models are evaluated on the same untouched test split.

📌 Evaluation Metrics
Metric	Description
Accuracy	Overall classification correctness
Precision	Reliability of positive predictions
Recall	Coverage of actual positive samples
F1 Score	Harmonic mean of precision and recall
Macro F1	Average F1 across both classes
🤖 Models Compared
                 ┌──────────────────────────┐
                 │     Test Dataset         │
                 └────────────┬─────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │ TF-IDF +    │ │   Frozen    │ │ Fine-Tuned  │
       │ Logistic    │ │    BERT     │ │    BERT     │
       │ Regression  │ │             │ │             │
       └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Model Comparison │
                    └──────────────────┘

Results are stored in:

experiments/model_comparison.csv
🔎 Error Analysis

The project performs per-example error analysis to understand where the model succeeds and fails.

For every test example, the following information is recorded:

Field	Description
sentence	Input text
actual_label	Ground-truth sentiment
predicted_label	Model prediction
confidence	Prediction confidence
correct	Whether prediction was correct

The analysis helps identify difficult cases involving:

❌ Negation
🔀 Mixed or contrasting sentiment
📏 Long text sequences
⚠️ High-confidence incorrect predictions

Output file:

results/error_analysis.csv
🎲 Reproducibility

The project uses a fixed random seed:

Seed = 42

The seed is applied to:

Python random operations
NumPy
PyTorch CPU operations
PyTorch CUDA operations
Dataset splitting

This ensures consistent dataset splits and improves experiment reproducibility.

⚡ Compute & Memory Optimization

The training pipeline is designed to work across different hardware configurations.

🖥️ Hardware-Aware Training
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
⚙️ Optimization Techniques
⚡ CUDA mixed precision / FP16
📦 Gradient accumulation
🔢 Configurable batch sizes
📊 Configurable evaluation batch sizes
✂️ Gradient clipping
📏 Dynamic padding
🚀 Dynamic sequence-length selection
📏 Dynamic Sequence Length

Instead of automatically using BERT's maximum sequence length of 512, the pipeline analyzes the token-length distribution of the training dataset.

An appropriate sequence length is selected based on the observed data distribution.

Benefits
⚡ Faster training
💾 Lower memory consumption
📉 Less unnecessary padding
🚀 More efficient inference
📂 Project Structure
bert-sentiment-analysis/
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
🛠️ Tech Stack
Category	Technologies
🐍 Programming	Python 3.9+
🤖 Deep Learning	PyTorch
🧠 NLP	BERT, Hugging Face Transformers
📊 Machine Learning	Scikit-learn
🧹 Data Processing	Pandas, NumPy
📈 Visualization	Matplotlib, Seaborn
🌐 Deployment	Streamlit
📓 Experimentation	Jupyter Notebook
⚙️ Installation
1️⃣ Clone the Repository
git clone <your-repository-url>
cd bert-sentiment-analysis
2️⃣ Create a Virtual Environment
🍎 macOS / 🐧 Linux
python -m venv .venv
source .venv/bin/activate
🪟 Windows
python -m venv .venv
.venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
🗃️ Dataset Setup

Place the dataset in:

data/
└── sentiment_train.csv

The CSV file should contain the following columns:

sentence
label

The data pipeline automatically validates:

📄 File existence
🏷️ Required columns
❓ Missing values
♻️ Duplicate records
⚖️ Label distribution
📝 Text length
🔤 Token length
🚀 Training Pipeline
1️⃣ Validate the Dataset
python -m src.data
2️⃣ Train TF-IDF Baseline
python -m src.baseline

The trained model is saved under:

models/tfidf_logistic/
3️⃣ Train Frozen BERT
python -m src.train \
    --freeze_bert \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/frozen_bert
4️⃣ Run Learning-Rate Search
python -m src.train \
    --mode lr_search \
    --epochs 3

Results:

experiments/learning_rate_results.csv
5️⃣ Run Epoch Search

After selecting the preferred learning rate:

python -m src.train \
    --mode epoch_search \
    --learning_rate 2e-5

Results:

experiments/epoch_results.csv
6️⃣ Train Fully Fine-Tuned BERT
python -m src.train \
    --epochs 4 \
    --learning_rate 2e-5 \
    --output_dir models/best_bert
📊 Model Evaluation
🔬 Compare All Models
python -m src.evaluate --compare_all

The evaluation compares:

TF-IDF + Logistic Regression
Frozen BERT
Fully Fine-Tuned BERT

Results are saved to:

experiments/model_comparison.csv
🧪 Evaluate the Final BERT Model
python -m src.evaluate \
    --model_dir models/best_bert
🔍 Error Analysis

Run:

python -m src.error_analysis \
    --model_dir models/best_bert

Output:

results/error_analysis.csv
🌐 Streamlit Application

The project includes an interactive web application for real-time sentiment prediction.

▶️ Launch the Application
streamlit run app/streamlit_app.py

The application allows users to:

✍️ Enter an English sentence
🧠 Generate a sentiment prediction
📊 View prediction confidence
🔄 Compare available models
Example
Input:
"This movie was absolutely fantastic."

Prediction:
✅ POSITIVE

Note: Prediction confidence depends on the trained model and input text.

🏭 Deployment Architecture
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
💡 Key Engineering Decisions
1️⃣ Baseline Before Deep Learning

A TF-IDF + Logistic Regression model establishes a simple and interpretable benchmark before introducing transformer architectures.

2️⃣ Frozen vs Fine-Tuned BERT

Both configurations are evaluated to distinguish the value of pretrained representations from the additional benefit of task-specific fine-tuning.

3️⃣ Validation-Based Model Selection

Hyperparameters are selected using validation performance while the test set remains isolated for final evaluation.

4️⃣ F1-Based Early Stopping

Validation F1 is used for checkpoint selection to provide a balanced measure of classification performance.

5️⃣ Reproducible Experiments

Fixed seeds and consistent data splitting improve experiment reproducibility and comparison.

6️⃣ Production-Oriented Inference

The Streamlit application loads trained model checkpoints and performs inference without retraining the models during application startup.

📈 Results

Model performance is generated automatically during evaluation.

Model	Accuracy	Precision	Recall	F1 Score	Macro F1
📊 TF-IDF + Logistic Regression	—	—	—	—	—
🧊 Frozen BERT	—	—	—	—	—
🔥 Fine-Tuned BERT	—	—	—	—	—

📌 Note: Run the training and evaluation pipeline to populate the table with the actual experiment results.

🔮 Future Improvements
🤏 DistilBERT for faster inference
🚀 ONNX / TorchScript optimization
🌐 FastAPI inference API
🐳 Dockerized deployment
📊 MLflow / Weights & Biases experiment tracking
⚖️ Class-weighted training
🎯 Probability calibration
📦 Batch inference support
🔄 Automated CI/CD testing
🧪 Robustness and adversarial testing
📜 License

This project is intended for educational, research, and portfolio purposes.

If distributing the project publicly, consider adding an appropriate open-source license such as the MIT License.

👨‍💻 Author
Saket Pandey

🎓 IIT Madras
🧬 Biotechnology / Computational Biology
