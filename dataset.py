import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TicketDataset(Dataset):
    def __init__(self, texts, labels, model_name="xlm-roberta-large", max_len=128):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label_data = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tox_labels': torch.tensor(label_data['toxic'], dtype=torch.float),
            'intent_labels': torch.tensor(label_data['intents'], dtype=torch.float)
        }