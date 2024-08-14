import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_length = max_length
        self.examples = self._load_data()

    def _load_data(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        encodings = [self.tokenizer.encode(line, truncation=True, max_length=self.max_length) for line in lines]
        return encodings

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoding = self.examples[idx]
        item = {'input_ids': torch.tensor(encoding, dtype=torch.long)}
        item['labels'] = item['input_ids'].clone()
        return item

