"""
Central configuration for the BERT Sentiment Analysis project.

Every path, hyperparameter default, and constant used across the project
lives here so that scripts, notebooks, and the Streamlit app all agree.

Nothing in this file trains a model or touches the dataset -- it only
declares configuration values and lightweight dataclasses.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# --------------------------------------------------------------------------
# Paths (all relative to the project root)
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "sentiment_train.csv")

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TFIDF_MODEL_DIR = os.path.join(MODELS_DIR, "tfidf_logistic")
FROZEN_BERT_DIR = os.path.join(MODELS_DIR, "frozen_bert")
BEST_BERT_DIR = os.path.join(MODELS_DIR, "best_bert")

LR_RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "learning_rate_results.csv")
EPOCH_RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "epoch_results.csv")
MODEL_COMPARISON_PATH = os.path.join(EXPERIMENTS_DIR, "model_comparison.csv")

CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
MODEL_COMPARISON_PLOT_PATH = os.path.join(RESULTS_DIR, "model_comparison.png")
TRAINING_CURVES_PATH = os.path.join(RESULTS_DIR, "training_curves.png")
ERROR_ANALYSIS_PATH = os.path.join(RESULTS_DIR, "error_analysis.csv")

# --------------------------------------------------------------------------
# Dataset columns / constants
# --------------------------------------------------------------------------
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"

# Verified against the downloaded dataset by src/data.py::summarize_dataset().
# Do not change this blindly -- see README for the verification note.
LABEL_MAP = {0: "Negative", 1: "Positive"}

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Splits (test set is never touched until final evaluation)
# --------------------------------------------------------------------------
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority:
        1. CUDA GPU
        2. Apple Silicon MPS GPU
        3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def get_device_name() -> str:
    """Return a human-readable name for the selected device."""
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)

    if DEVICE.type == "mps":
        return "Apple Silicon GPU (MPS)"

    return "CPU"


print(f"Using device: {DEVICE}")
print(f"Device name: {get_device_name()}")

USE_FP16 = DEVICE.type == "cuda"
# --------------------------------------------------------------------------
# Training configuration (all overridable from CLI in train.py)
# --------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_name: str = BERT_MODEL_NAME
    max_length: Optional[int] = None          # resolved from data if None
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    early_stopping_metric: str = "val_f1"      # maximize
    freeze_bert: bool = False
    seed: int = SEED
    fp16: bool = USE_FP16
    output_dir: str = BEST_BERT_DIR

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_TRAINING_CONFIG = TrainingConfig()

LEARNING_RATES_TO_SEARCH = [1e-5, 2e-5, 3e-5, 5e-5]
EPOCHS_TO_SEARCH = [2, 3, 4, 5]
