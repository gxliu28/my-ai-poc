import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from transformers import GPT2Tokenizer
from dataset.text_dataset import TextDataset

@pytest.fixture
def sample_dataset():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return TextDataset(tokenizer, 'test/test_sample.txt', max_length=10)

def test_text_dataset_len(sample_dataset):
    assert len(sample_dataset) == 8  # Assuming test_sample.txt has 8 line

def test_text_dataset_item(sample_dataset):
    item = sample_dataset[0]
    assert 'input_ids' in item
    assert 'labels' in item
    assert len(item['input_ids']) <= 10
    assert len(item['labels']) <= 10

