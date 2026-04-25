import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

def run_evaluation(model, data_loader, device, lang_list=None):
    model.eval()
    all_tox_preds = []
    all_tox_probs = []
    all_tox_labels = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            tox_labels = d["tox_labels"].to(device)

            # Use the correct amp syntax for 2026/latest torch
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                tox_logits, _ = model(input_ids, attention_mask)
            
            # Cast to float32 before numpy conversion to avoid ScalarType error
            probs = torch.sigmoid(tox_logits).to(torch.float32).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_tox_preds.extend(preds)
            all_tox_probs.extend(probs)
            all_tox_labels.extend(tox_labels.cpu().numpy().flatten())

    # Calculate Metrics
    report = classification_report(all_tox_labels, all_tox_preds, target_names=['Clean', 'Toxic'])
    auc = roc_auc_score(all_tox_labels, all_tox_probs)
    cm = confusion_matrix(all_tox_labels, all_tox_preds)
    
    lang_results = {}
    if lang_list is not None:
        df_results = pd.DataFrame({
            'label': all_tox_labels,
            'pred': all_tox_preds,
            'lang': lang_list
        })
        for lang in df_results['lang'].unique():
            lang_df = df_results[df_results['lang'] == lang]
            # Use macro F1 to account for class imbalance in specific languages
            f1 = f1_score(lang_df['label'], lang_df['pred'], average='macro')
            lang_results[lang] = f1
            
    return report, auc, lang_results, cm

def run_intent_evaluation(model, data_loader, device):
    """
    Evaluate per-intent ROC-AUC scores on the English test set.
    
    Returns:
        Dictionary with intent names as keys and ROC-AUC scores as values
    """
    model.eval()
    intent_names = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    all_intent_probs = []
    all_intent_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating Intents"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            intent_labels = d["intent_labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, intent_logits = model(input_ids, attention_mask)
            
            # Cast to float32 BEFORE sigmoid for numerical stability
            intent_probs = torch.sigmoid(intent_logits.float()).cpu().numpy()
            all_intent_probs.append(intent_probs)
            all_intent_labels.append(intent_labels.cpu().numpy())

    # Concatenate all batches
    all_intent_probs = np.vstack(all_intent_probs)
    all_intent_labels = np.vstack(all_intent_labels)

    # Calculate per-intent ROC-AUC
    intent_auc_scores = {}
    
    print("\n--- PER-INTENT ROC-AUC SCORES ---")
    for i, name in enumerate(intent_names):
        y_true = all_intent_labels[:, i]
        y_pred = all_intent_probs[:, i]
        
        # Only evaluate on valid labels (0 or 1, not -1)
        mask = (y_true == 0) | (y_true == 1)
        
        if len(np.unique(y_true[mask])) > 1:  # Need both classes to compute AUC
            auc = roc_auc_score(y_true[mask], y_pred[mask])
            intent_auc_scores[name] = auc
            print(f"{name.upper():<20}: {auc:.4f}")
        else:
            print(f"{name.upper():<20}: N/A (only one class in data)")
            intent_auc_scores[name] = None

    return intent_auc_scores