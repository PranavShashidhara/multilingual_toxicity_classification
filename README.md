# Multilingual Toxicity Classification with Intent Detection

A multi-task NLP system for detecting toxic content and identifying harmful intent across 15+ languages using transformer-based models.

## Project Overview

This project builds a system that:
1. **Binary Toxicity Classification**: Detects toxic vs. non-toxic content
2. **Multi-Label Intent Detection**: Identifies harmful intent categories (Insult, Threat, Obscenity, Identity Hate)
3. **Multilingual Support**: Works across 15+ languages with a single model

### Architecture

- **Backbone**: XLM-RoBERTa-Large (560M parameters)
- **Multi-Task Learning**: Shared backbone with dual classification heads (toxicity + intent)
- **Training**: Mixed-precision (bfloat16) with gradient scaling and linear LR warmup

---

## Results

### English (Jigsaw Test Set — 5,000 samples)

| Metric | Value |
|--------|-------|
| Accuracy | 96% |
| AUC-ROC | **0.9867** |
| Clean F1 | 0.98 |
| Toxic F1 | 0.81 |
| Macro F1 | 0.90 |

Confusion matrix: 4,432 true negatives / 390 true positives / 98 false positives / 80 false negatives.

### Multilingual (TextDetox Test Set — 13,275 samples, 15+ languages)

| Metric | Value |
|--------|-------|
| Accuracy | 87% |
| AUC-ROC | **0.9460** |
| Macro F1 | 0.87 |
| Weighted F1 | 0.87 |

### Per-Language F1 Scores

| Language | F1 |
|----------|----|
| ZH (Chinese) | 0.7591 |
| HIN | 0.7612 |
| HE (Hebrew) | 0.7820 |
| AR (Arabic) | 0.7969 |
| AM (Amharic) | 0.8253 |
| IT (Italian) | 0.8494 |
| TT (Tatar) | 0.8724 |
| JA (Japanese) | 0.8783 |
| DE (German) | 0.8816 |
| ES (Spanish) | 0.8905 |
| UK (Ukrainian) | 0.9283 |
| HI (Hindi) | 0.9294 |

> **Note:** Lower F1 on ZH and HIN reflects the cross-script generalization challenge; these are targets for future improvement.

---

## Datasets

### 1. Jigsaw Toxic Comment Classification (English)

**Source**: [thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)

**Labels**: Binary (`toxic`), Multi-label (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)

### 2. Multilingual Toxicity Dataset

**Source**: [textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)

**Languages**: Arabic, Russian, Chinese, Spanish, German, French, Italian, Portuguese, Hindi, Ukrainian, Turkish, Tatar, Kazakh, and others (14+ non-English languages)

**Labels**: Binary toxicity (`toxic`)

---

## Project Structure

```
multilingual_toxicity_classification/
├── preprocess.py
├── dataset.py
├── model_utils.py
├── eval.py
├── validate.py
├── model_train.ipynb
├── model_inference_evaluationl.ipynb
├── upload_model_to_hugging_face.ipynb
├── requirements.txt
├── setup.sh
└── data/
    └── processed/
        ├── jigsaw/
        │   ├── jigsaw_train.csv
        │   ├── jigsaw_val.csv
        │   └── jigsaw_test.csv
        └── multilingual_toxic/
            ├── merged_non_en_train.csv
            ├── merged_non_en_test.csv
            └── per_language_test/
```

## File Descriptions

| File | Description |
|------|-------------|
| `preprocess.py` | Loads Jigsaw and TextDetox datasets, cleans text, saves stratified train/val/test CSVs. Jigsaw split 90/10; multilingual 80/20 with per-language test sets. |
| `dataset.py` | `TicketDataset` — PyTorch Dataset that tokenizes text and returns `input_ids`, `attention_mask`, `tox_labels`, `intent_labels`. |
| `model_utils.py` | `MultiTaskXLMR` model with shared backbone and separate toxicity/intent heads. Provides `train_epoch` (English, weighted multi-task loss) and `train_epoch_multilingual` (masked intent loss). |
| `eval.py` | Computes classification report, ROC-AUC, and per-language macro F1. |
| `validate.py` | Per-intent ROC-AUC validation pass used during training. |
| `model_train.ipynb` | End-to-end training: 3 epochs on Jigsaw (batch 64, AdamW, bfloat16), then 2-epoch multilingual fine-tuning at LR 5e-6. |
| `model_inference_evaluationl.ipynb` | Downloads models from HF Hub, evaluates on test sets, runs direct inference examples. |
| `upload_model_to_hugging_face.ipynb` | Uploads model weights and cards to Hugging Face Hub. |

---

## Trained Models

Both models are publicly available on Hugging Face Hub:

- **English**: [`pshashid/xlmr-toxicity-english`](https://huggingface.co/pshashid/xlmr-toxicity-english) — trained on Jigsaw (toxicity + intent heads), **0.9867 AUC-ROC**
- **Multilingual**: [`pshashid/xlmr-toxicity-multilingual`](https://huggingface.co/pshashid/xlmr-toxicity-multilingual) — fine-tuned on 14+ languages, **0.9460 AUC-ROC**

---

## Setup

```bash
git clone https://github.com/PranavShashidhara/multilingual_toxicity_classification.git
cd multilingual_toxicity_classification
bash setup.sh
source venv/bin/activate
```

## Usage

```bash
# 1. Preprocess
python preprocess.py

# 2. Train — open and run model_train.ipynb sequentially

# 3. Evaluate — open model_inference_evaluationl.ipynb

# 4. (Optional) Upload to HF Hub — run upload_model_to_hugging_face.ipynb
```

## Technical Stack

- **ML Framework**: PyTorch (mixed-precision bfloat16)
- **Models**: XLM-RoBERTa-Large via Hugging Face Transformers
- **Data**: Hugging Face Datasets, Pandas, scikit-learn
- **Monitoring**: TensorBoard, Weights & Biases
- **Model Hosting**: Hugging Face Hub