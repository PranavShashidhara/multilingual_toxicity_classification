# Multilingual Toxicity Classification with Intent Detection

A multi-task NLP system for detecting toxic content and identifying harmful intent across 15+ languages using transformer-based models.

## Project Overview

This project aims to build a system that:
1. **Binary Toxicity Classification**: Detects toxic vs. non-toxic content
2. **Multi-Label Intent Detection**: Identifies harmful intent categories (Insult, Threat, Obscenity, Identity Hate)
3. **Multilingual Support**: Works across 15+ languages with a single model

### Planned Architecture

- **Backbone**: XLM-RoBERTa-Large (560M parameters)
- **Multi-Task Learning**: Shared backbone with dual classification heads
- **Training**: PyTorch DDP for multi-GPU efficiency

## Datasets

### 1. Jigsaw Toxic Comment Classification (English)

**Source**: [thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)

**Labels**: 
- Binary: `toxic`
- Multi-label: `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

### 2. Multilingual Toxicity Dataset

**Source**: [textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)

**Languages**: Arabic, Russian, Chinese, Spanish, German, French, Italian, Portuguese, Hindi, Ukrainian, Turkish, Tatar, Kazakh, and others (14+ non-English languages)

**Labels**: Binary toxicity (`toxic`)

## Setup

### Installation
```bash
# Clone repository
git clone https://github.com/PranavShashidhara/multilingual_toxicity_classification.git
cd multilingual_toxicity_classification

# Set up environment
bash setup.sh

# Activate environment
source venv/bin/activate
```

## Data Preprocessing

### Running Preprocessing
```bash
python preprocess.py
```

### What It Does

**Jigsaw Dataset:**
- Loads English toxic comment data from Hugging Face
- Cleans text (removes newlines, strips whitespace)
- Splits into train (90%) and validation (10%) with stratification
- Saves to `data/processed/jigsaw/`

**Multilingual Dataset:**
- Loads 14+ non-English languages
- Merges all non-English data into single training set
- Splits 80/20 train/test with stratification by language
- Saves merged data to `data/processed/multilingual_toxic/`
- Also saves per-language test sets in `per_language_test/`

### Output Structure
```
data/processed/
├── jigsaw/
│   ├── jigsaw_train.csv
│   ├── jigsaw_val.csv
│   └── jigsaw_test.csv
└── multilingual_toxic/
    ├── merged_non_en_train.csv
    ├── merged_non_en_test.csv
    └── per_language_test/
        ├── ar_test.csv
        ├── ru_test.csv
        ├── zh_test.csv
        └── ...
```



## Current Status

✅ **Completed:**
- Data preprocessing pipeline
- Stratified train/val/test splits
- Per-language test set generation
- Text cleaning and normalization

🚧 **Next Steps:**
- Model architecture implementation
- Training pipeline with DDP
- Evaluation metrics (Macro-F1, ROC-AUC, Hamming Loss, mAP)
- Fairness analysis (Subgroup AUC, bias metrics)


## Technical Stack

- **Data**: Hugging Face Datasets, Pandas
- **ML Framework**: PyTorch (planned)
- **Models**: Transformers (planned)
- **Preprocessing**: scikit-learn
