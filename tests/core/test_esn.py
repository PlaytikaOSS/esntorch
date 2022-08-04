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
# pytest tests/core/test_esn.py
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

    # Device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # global

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
        dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer))

    return dataset_d, dataloader_d


def instantiate_esn(**kwargs):
    """Train ESN with a designated learning algorithm."""

    # ESN parameters
    esn_params = {
        'embedding': 'bert-base-uncased',
        'dim': 1000,
        'bias_scaling': 0.1,
        'sparsity': 0.,
        'spectral_radius': None,
        'leaking_rate': 0.17647315261153904,
        'activation_function': 'tanh',
        'input_scaling': 0.1,
        'mean': 0.0,
        'std': 1.0,
        'learning_algo': None,  # initialzed below
        'criterion': None,  # initialzed below
        'optimizer': None,  # initialzed below
        'pooling_strategy': 'mean',
        'bidirectional': False,
        'device': device,
        'seed': 42345,
        **kwargs
    }

    # Instantiate the ESN
    ESN = esn.EchoStateNetwork(**esn_params)

    # Define the learning algo of the ESN
    ESN.learning_algo = la.RidgeRegression(alpha=7.843536845714804)

    # Put the ESN on the device (CPU or GPU)
    ESN = ESN.to(device)

    return ESN


def warm_up(ESN, dataset_d):
    """Warm up ESN."""

    if isinstance(ESN.layer, res.LayerRecurrent):
        ESN.warm_up(dataset_d['train'].select(range(10)))


def predict_esn(ESN, dataloader_d):
    """Compute predictions and accuracy."""

    # Train predictions and accuracy
    print("Predict on train set")
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    #  train_acc = train_acc.item() if ESN.device.type == 'cuda' else train_acc

    # Test predictions and accuracy
    print("Predict on test set")
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    #  test_acc = test_acc.item() if ESN.device.type == 'cuda' else test_acc

    return train_pred, train_acc, test_pred, test_acc


def test_warm_up(create_dataset):
    """Test warm_up method."""

    dataset_d, dataloader_d = create_dataset
    mode, distribution = 'recurrent_layer', 'uniform'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    initial_state = ESN.layer.initial_state
    warm_up(ESN, dataset_d)
    warm_state = ESN.layer.initial_state
    assert (initial_state != warm_state).any()


def test_fit_and_predict(create_dataset):
    """Test fit and predict methods"""

    def test(**kwargs):
        if kwargs['deep']:
            la_input_dim = kwargs['nb_layers'] * kwargs['dim']
        else:
            la_input_dim = kwargs['dim']

        dataset_d, dataloader_d = create_dataset
        ESN = instantiate_esn(**kwargs)

        # test fit via _fit_direct
        ESN.learning_algo = la.RidgeRegression(alpha=7.843536845714804)
        weights_before_fit = ESN.learning_algo.weights
        ESN = ESN.to(device)
        assert weights_before_fit is None
        ESN.fit(dataloader_d["train"])
        weights_after_fit = ESN.learning_algo.weights
        assert torch.is_tensor(weights_after_fit)

        # test predict
        train_pred, train_acc, test_pred, test_acc = predict_esn(ESN, dataloader_d)
        assert len(train_pred) == len(dataset_d['train'])
        assert len(test_pred) == len(dataset_d['test'])
        assert test_acc > 0.8

        # test fit via _fit_GD
        ESN.learning_algo = la.LogisticRegression(input_dim=la_input_dim, output_dim=6)
        weights_before_fit = ESN.learning_algo.linear.weight.clone().to(device)
        ESN.criterion = torch.nn.CrossEntropyLoss()
        ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)
        ESN = ESN.to(device)
        ESN.fit(dataloader_d["train"], epochs=1, iter_steps=10)
        weights_after_fit = ESN.learning_algo.linear.weight
        assert (weights_before_fit != weights_after_fit).any()

        # test predict
        train_pred, train_acc, test_pred, test_acc = predict_esn(ESN, dataloader_d)
        assert len(train_pred) == len(dataset_d['train'])
        assert len(test_pred) == len(dataset_d['test'])
        assert test_acc > 0.8

    test(deep=False, mode='recurrent_layer', distribution='uniform', dim=300)
    test(deep=True, nb_layers=3, mode='recurrent_layer', distribution='uniform', dim=300)
