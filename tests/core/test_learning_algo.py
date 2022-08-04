# The MIT License (MIT)
#
# Copyright (c) 2021 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# *** INSTRUCTIONS ***
# For the test, the following packages need to be installed:
# pip install pytest
# pip install pytest-cov
# ---------------
# Also, use torch version 1.7.1: some functions do not work with torch version 1.9.0
# ---------------
# To launch this test file only, run the following command:
# pytest tests/core/test_learning_algo.py
# To launch all tests inside /esntorch/core/ with line coverage, run the following command:
# pytest --cov tests/core/
# *** END INSTRUCTIONS ***

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
import esntorch.core.learning_algo as la
import esntorch.core.reservoir as res
import esntorch.core.esn as esn
import pytest


# We train an ESN with the different learning algorithms on 20% of the TREC dataset.
# We test whether the train and test perfromance is higher than 70%.

@pytest.fixture()
def create_dataset():
    """Preliminary function to create dataset required for the tests."""

    # Load and Tokenize Data
    def tokenize(sample):
        """Tokenize sample"""

        sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)

        return sample

    def load_and_prepare_dataset(dataset_name, split, cache_dir):
        """
        Load dataset from the datasets library of HuggingFace.
        Tokenize and add length.
        """

        # Load dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
        if isinstance(dataset, list):
            dataset = DatasetDict({"train": dataset[0], "test": dataset[1]})

        # Rename label column for tokenization purposes
        dataset = dataset.rename_column('label-coarse', 'labels')

        # Tokenize data
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column('length', 'lengths')
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'lengths'])

        return dataset

    # Load BERT tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load and prepare data
    CACHE_DIR = 'cache_dir/'
    dataset = load_and_prepare_dataset('trec', split=['train[:20%]', 'test[:10%]'], cache_dir=CACHE_DIR)
    train_dataset = dataset['train'].sort("lengths")
    test_dataset = dataset['test'].sort("lengths")
    # Create dict of all datasets
    dataset_d = {
        'train': train_dataset,
        'test': test_dataset
    }

    # Create dataloaders
    dataloader_d = {}
    for k, v in dataset_d.items():
        dataloader_d[k] = torch.utils.data.DataLoader(v,
                                                      batch_size=256,
                                                      collate_fn=DataCollatorWithPadding(tokenizer))

    return dataset_d, dataloader_d


def train_esn(dataset_d, dataloader_d, learning_algo=None, bidirectional=False):
    """Train ESN with a designated learning algorithm."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ESN parameters
    esn_params = {
        'embedding': 'bert-base-uncased',
        'distribution': 'uniform',  # uniform, gaussian
        'dim': 1000,
        'bias_scaling': 0.,
        'sparsity': 0.,
        'spectral_radius': 0.7094538192983408,
        'leaking_rate': 0.17647315261153904,
        'activation_function': 'relu',  # 'tanh', 'relu'
        'input_scaling': 0.1,
        'mean': 0.0,
        'std': 1.0,
        'learning_algo': None,  # initialzed below
        'criterion': None,  # initialzed below
        'optimizer': None,  # initialzed below
        'pooling_strategy': 'mean',  # 'mean', 'last', None
        'bidirectional': bidirectional,
        'device': device,
        'mode': 'recurrent_layer',  # 'no_layer, 'linear_layer', 'recurrent_layer'
        'seed': 42
    }

    # Instantiate the ESN
    ESN = esn.EchoStateNetwork(**esn_params)

    # Define the learning algo of the ESN
    if bidirectional:
        input_dim = esn_params['dim'] * 2
    else:
        input_dim = esn_params['dim']

    if learning_algo == 'ridge':
        ESN.learning_algo = la.RidgeRegression(alpha=7.843536845714804)
    elif learning_algo == 'ridge_skl':
        ESN.learning_algo = la.RidgeRegression_skl(alpha=7.843536845714804)
    elif learning_algo == 'svc':
        ESN.learning_algo = la.LinearSVC()
    elif learning_algo == 'logistic':
        ESN.learning_algo = la.LogisticRegression(input_dim=input_dim, output_dim=6)
        ESN.criterion = torch.nn.CrossEntropyLoss()
        ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)
    elif learning_algo == 'logistic_skl':
        ESN.learning_algo = la.LogisticRegression_skl()
    elif learning_algo == 'deep_nn':
        ESN.learning_algo = la.DeepNN([input_dim, 512, 256, 6])
        ESN.criterion = torch.nn.CrossEntropyLoss()
        ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)

    # Put the ESN on the device (CPU or GPU)
    ESN = ESN.to(device)

    # Warm up ESN if necessary
    if isinstance(ESN.layer, res.LayerRecurrent):
        ESN.warm_up(dataset_d['train'].select(range(10)))

    # Training ESN
    ESN.fit(dataloader_d["train"], epochs=3, iter_steps=10)  # Parameter epochs used only with LogisticRegression

    # Results
    # Train predictions and accuracy
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    #  train_acc = train_acc.item() if device.type == 'cuda' else train_acc # XXX

    # Test predictions and accuracy
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    # test_acc = test_acc.item() if device.type == 'cuda' else test_acc # XXX

    return train_acc, test_acc


def test_RidgeRegression_fit(create_dataset):
    """Test RidgeRegression."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='ridge', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8


def test_RidgeRegression_skl_fit(create_dataset):
    """Test RidgeRegression_skl."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='ridge_skl', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8


def test_LogisticRegression_fit(create_dataset):
    """Test LogisticRegression."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='logistic', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8


def test_LogisticRegression_skl_fit(create_dataset):
    """Test LogisticRegression_skl."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='logistic_skl', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8


def test_LinearSVC_fit(create_dataset):
    """Test LinearSVC."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='svc', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8


def test_DeepNN_fit(create_dataset):
    """Test DeepNN."""

    dataset_d, dataloader_d = create_dataset
    for b in [False, True]:
        train_acc, test_acc = train_esn(dataset_d=dataset_d, dataloader_d=dataloader_d,
                                        learning_algo='deep_nn', bidirectional=b)
        assert train_acc > 0.8 and test_acc > 0.8
