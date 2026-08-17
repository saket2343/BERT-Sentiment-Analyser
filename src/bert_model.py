"""
BERT-based classifier used for both Model 2 (Frozen BERT) and
Model 3 (Fully Fine-Tuned BERT). The only difference between the two is
whether the BERT encoder's parameters are frozen -- controlled by the
`freeze_bert` flag, so there is a single, shared, well-tested architecture.

Architecture:
    English Text -> BERT Tokenizer -> BERT Encoder -> CLS representation
    -> Dropout -> Linear classification head -> Binary Sentiment
"""

import json
import os

import torch
from torch import nn
from transformers import AutoModel, BertTokenizerFast

from src.config import BERT_MODEL_NAME, NUM_LABELS, LABEL_MAP


class BertSentimentClassifier(nn.Module):
    """CLS-token classification head on top of a BERT encoder.

    Set `freeze_bert=True` for Model 2 (Frozen BERT): only the
    classification head is trained.
    Set `freeze_bert=False` for Model 3 (Fully Fine-Tuned BERT): all BERT
    parameters are trainable.
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.2,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager"
)
        hidden_size = self.bert.config.hidden_size

        self.freeze_bert = freeze_bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation: first token of the last hidden state.
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_model(
    model: BertSentimentClassifier,
    tokenizer: BertTokenizerFast,
    output_dir: str,
    max_length: int,
    extra_config: dict = None,
):
    """Save weights, tokenizer, config, and label mapping so the model can
    be reloaded without retraining (used by evaluate.py and the Streamlit app).
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model_state_dict.pt"))
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_name": model.bert.config.name_or_path,
        "num_labels": model.classifier.out_features,
        "dropout": model.dropout.p,
        "freeze_bert": model.freeze_bert,
        "max_length": max_length,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
    }
    if extra_config:
        config.update(extra_config)

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {output_dir}")


def load_model(model_dir: str, device: torch.device = None):
    """Reusable model-loading function.

    Returns (model, tokenizer, config). Does NOT train anything.
    """
    config_path = os.path.join(model_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No training_config.json found in {model_dir}. "
            f"Train and save a model first (see README)."
        )

    with open(config_path) as f:
        config = json.load(f)

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)

    model = BertSentimentClassifier(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        dropout=config["dropout"],
        freeze_bert=config["freeze_bert"],
    )
    state_dict_path = os.path.join(model_dir, "model_state_dict.pt")
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    if device is not None:
        model.to(device)
    model.eval()

    return model, tokenizer, config
