
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np 
import torch
def validate_intents(model, data_loader, device):
    model.eval()
    intent_names = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["intent_labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, intent_logits = model(input_ids, attention_mask)
            
            # CRITICAL: Cast to float32 BEFORE sigmoid to keep precision
            probs = torch.sigmoid(intent_logits.float()).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\n--- DEBUG: First 5 Rows (Post-Fix) ---")
    for i in range(5):
        # Only show labels that aren't -1 or nan
        print(f"Sample {i} Labels: {all_labels[i]}")
        print(f"Sample {i} Preds : {all_preds[i]}")

    print("\n--- FINAL PER-INTENT ROC-AUC ---")
    for i, name in enumerate(intent_names):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        
        # Only evaluate on 0s and 1s
        mask = (y_true == 0) | (y_true == 1)
        
        if len(np.unique(y_true[mask])) > 1:
            score = roc_auc_score(y_true[mask], y_pred[mask])
            print(f"{name.upper():<15} : {score:.4f}")