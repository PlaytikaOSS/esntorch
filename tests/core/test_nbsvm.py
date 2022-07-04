import pytest
from datasets import load_dataset, Dataset, concatenate_datasets

from transformers import AutoTokenizer
from esntorch.core.nbsvm import *


def test_prepare_datasets():
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(sample):
        """Tokenize sample"""

        sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)

        return sample

    dataset = load_dataset('imdb', split='train[:100]+test[:100]', cache_dir='cache_dir/')
    # Rename label column for tokenization purposes
    dataset = dataset.rename_column('label', 'labels')
    # Tokenize data
    dataset = dataset.map(tokenize, batched=True),
    dataset = dataset.rename_column('length', 'lengths'),
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'lengths'])
    n_train, n_test = len(dataset['train']), len(dataset['test'])
    prepare_datasets(dataset['train'], dataset['test'])

    assert len(dataset['train'][1]) == n_train and len(dataset['test'][1]) == n_test
    assert all(isinstance(item, int) for data in ('train', 'test') for item in dataset[data][1])
