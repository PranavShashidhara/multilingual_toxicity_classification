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

## Project Structure

```
multilingual_toxicity_classification/
├── preprocess.py                        # Loads, cleans, and splits Jigsaw + multilingual datasets into train/val/test CSVs
├── dataset.py                           # PyTorch Dataset class that tokenizes text and packages toxicity + intent labels
├── model_utils.py                       # Defines MultiTaskXLMR model and train_epoch / train_epoch_multilingual training loops
├── eval.py                              # Runs inference and computes classification report, ROC-AUC, and per-language F1
├── validate.py                          # Computes per-intent ROC-AUC scores on the validation set with masking for missing labels
├── model_train.ipynb                    # End-to-end training notebook: preprocessing → English training → multilingual fine-tuning
├── model_inference_evaluationl.ipynb    # Full evaluation pipeline: loads models from HF Hub → evaluates → runs direct inference examples
├── upload_model_to_hugging_face.ipynb   # Uploads trained model weights and model cards to Hugging Face Hub
├── requirements.txt                     # Python dependencies
├── setup.sh                             # Environment setup script
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
                ├── ar_test.csv
                ├── ru_test.csv
                ├── zh_test.csv
                └── ...
```

## File Descriptions

| File | Description |
|------|-------------|
| `preprocess.py` | Loads the Jigsaw (English) and TextDetox (multilingual) datasets from Hugging Face, cleans text, and saves stratified train/val/test CSVs. Jigsaw is split 90/10; multilingual data is split 80/20 with per-language test sets. |
| `dataset.py` | `TicketDataset` — PyTorch `Dataset` that tokenizes text with XLM-RoBERTa and returns `input_ids`, `attention_mask`, `tox_labels`, and `intent_labels`. |
| `model_utils.py` | `MultiTaskXLMR` model with shared XLM-RoBERTa backbone and separate toxicity/intent linear heads. Also provides `train_epoch` (English, weighted multi-task loss) and `train_epoch_multilingual` (masked intent loss for samples without intent labels). |
| `eval.py` | `run_evaluation` computes classification report, ROC-AUC, and per-language macro F1. `run_intent_evaluation` computes per-intent ROC-AUC with masking for unlabeled (`-1`) samples. |
| `validate.py` | `validate_intents` — lightweight per-intent ROC-AUC validation pass used during training to monitor intent head performance. |
| `model_train.ipynb` | End-to-end training notebook. Runs preprocessing, trains the model for 3 epochs on the Jigsaw English dataset (batch size 64, AdamW, bfloat16), then fine-tunes for 2 epochs on the merged multilingual dataset at a lower LR (5e-6). Saves checkpoints to Google Drive. |
| `model_inference_evaluationl.ipynb` | Complete inference and evaluation pipeline. Downloads both trained models from Hugging Face Hub, evaluates toxicity (classification report, ROC-AUC, confusion matrix) and intents (per-intent ROC-AUC) on held-out test sets, and runs direct inference examples including a batch inference helper. |
| `upload_model_to_hugging_face.ipynb` | Uploads trained `.bin` model weights to the Hugging Face Hub under `pshashid/xlmr-toxicity-english` and `pshashid/xlmr-toxicity-multilingual`, and creates model cards for each. |
| `requirements.txt` | Python dependencies: `transformers`, `datasets`, `torch`, `pandas`, `scikit-learn`, `tqdm`, `wandb`, `sentencepiece`, `tokenizers`, `langdetect`, `pyyaml`. |
| `setup.sh` | Creates a Python virtual environment, installs PyTorch and `requirements.txt`, and sets up the `data/processed` and `checkpoints` directories. |

## Trained Models

Both trained models are publicly available on Hugging Face Hub:

- **English model**: `pshashid/xlmr-toxicity-english` — trained on Jigsaw (toxicity + intent heads)
- **Multilingual model**: `pshashid/xlmr-toxicity-multilingual` — fine-tuned on 14+ non-English languages (toxicity head)

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

## Usage

### 1. Preprocess Data
```bash
python preprocess.py
```

### 2. Train the Model
Open and run `model_train.ipynb` sequentially. It will:
- Preprocess and save all datasets
- Train on English Jigsaw data (3 epochs)
- Fine-tune on merged multilingual data (2 epochs)
- Save model checkpoints

### 3. Evaluate
Open `model_inference_evaluationl.ipynb` to:
- Download models directly from Hugging Face Hub
- Evaluate on English and multilingual test sets
- Run direct inference on sample text

### 4. Upload to Hugging Face (optional)
Use `upload_model_to_hugging_face.ipynb` to publish new model weights to the Hub.

## Data Preprocessing Details

**Jigsaw Dataset:**
- Loads English toxic comment data from Hugging Face
- Cleans text (removes newlines, strips whitespace)
- Splits into train (90%) and validation (10%) with stratification
- Saves to `data/processed/jigsaw/`

**Multilingual Dataset:**
- Loads 14+ non-English languages
- Merges all non-English data into a single training set
- Splits 80/20 train/test with stratification by language
- Saves merged data to `data/processed/multilingual_toxic/`
- Also saves per-language test sets in `per_language_test/`

## Technical Stack

- **Data**: Hugging Face Datasets, Pandas
- **ML Framework**: PyTorch (mixed-precision bfloat16)
- **Models**: XLM-RoBERTa-Large (via Hugging Face Transformers)
- **Preprocessing**: scikit-learn
- **Monitoring**: TensorBoard (optional, integrated in training loop)
- **Model Hosting**: Hugging Face Hub
