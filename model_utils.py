import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from tqdm import tqdm
import numpy as np

class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name="xlm-roberta-large", num_intents=5):
        super(MultiTaskXLMR, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.toxicity_head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.intent_head = nn.Linear(self.backbone.config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        pooled_output = self.dropout(outputs.pooler_output)
        tox_logits = self.toxicity_head(pooled_output)
        intent_logits = self.intent_head(pooled_output)
        return tox_logits, intent_logits

def train_epoch(model, data_loader, optimizer, device, scaler, scheduler, writer=None, epoch=0):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    
    # Create the progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, d in enumerate(pbar):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        tox_labels = d["tox_labels"].to(device).view(-1, 1)
        intent_labels = d["intent_labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tox_logits, intent_logits = model(input_ids, attention_mask)
            loss_tox = criterion(tox_logits, tox_labels)
            loss_intent = criterion(intent_logits, intent_labels)
            total_loss = (1.0 * loss_tox) + (0.5 * loss_intent)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        current_loss = total_loss.item()
        losses.append(current_loss)
        
        # Update live display
        pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        # Log to TensorBoard if provided
        if writer:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/Total', current_loss, global_step)
            writer.add_scalar('Loss/Toxicity', loss_tox.item(), global_step)
            writer.add_scalar('Loss/Intent', loss_intent.item(), global_step)

    return np.mean(losses)

def train_epoch_multilingual(model, data_loader, optimizer, device, scaler, scheduler):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none') # Change to 'none' to allow masking
    losses = []
    
    pbar = tqdm(data_loader, desc="Multilingual Tuning")
    for d in pbar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        tox_labels = d["tox_labels"].to(device).view(-1, 1)
        intent_labels = d["intent_labels"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tox_logits, intent_logits = model(input_ids, attention_mask)
            
            # 1. Toxicity Loss (Usually always present)
            loss_tox = criterion(tox_logits, tox_labels).mean()
            
            # 2. Intent Loss with Masking
            # Only calculate loss where labels are NOT -1
            mask = (intent_labels != -1).float()
            loss_intent = (criterion(intent_logits, intent_labels) * mask).sum() / (mask.sum() + 1e-6)
            
            # If a batch has zero intent labels, ignore intent loss
            if mask.sum() == 0:
                total_loss = loss_tox
            else:
                total_loss = (1.0 * loss_tox) + (0.5 * loss_intent)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        losses.append(total_loss.item())
        pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

    return np.mean(losses)