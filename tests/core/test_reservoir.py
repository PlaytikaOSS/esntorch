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
# pytest tests/core/test_reservoir.py
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
        dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer))

    return dataset_d, dataloader_d


def instantiate_esn(**kwargs):
    """Train ESN with a designated learning algorithm."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ESN parameters
    esn_params = {
        'embedding': 'bert-base-uncased',
        'dim': 1000,
        'bias_scaling': 0.,
        'sparsity': 0.,
        'spectral_radius': None,
        'leaking_rate': 0.17647315261153904,
        'activation_function': 'relu',
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


def is_uniform(layer):
    """Test uniform distribution."""

    weights = layer.layer_w.view(-1)
    # assert layer dim
    assert layer.layer_w.shape == (layer.dim, layer.dim)
    # assert uniform distribution
    for i in range(5):
        assert torch.sum(weights <= -1 + i * 2 / 5).cpu() == pytest.approx(i / 5 * layer.dim ** 2, 0.05)


def is_gaussian(layer):
    """Test Gaussian distribution."""

    weights = layer.layer_w.view(-1)
    assert torch.mean(weights).cpu() == pytest.approx(layer.mean, abs=0.05)
    assert torch.std(weights).cpu() == pytest.approx(layer.std, 0.05)


def test_UniformReservoir():
    """Test uniform reservoir."""

    mode, distribution = 'recurrent_layer', 'uniform'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    layer = ESN.layer
    is_uniform(layer)


def test_GaussianReservoir():
    """Test Gaussian reservoir."""

    mode, distribution = 'recurrent_layer', 'gaussian'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    layer = ESN.layer
    is_gaussian(layer)


def test_LayerLinear():
    """Test LayerLinear."""

    mode, distribution = 'linear_layer', 'uniform'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    reservoir = ESN.layer
    weights = vars(reservoir).get('layer_w', None)
    assert weights is None
    input_w = reservoir.input_w if hasattr(reservoir, 'input_w') else None
    assert input_w is not None
    assert input_w.shape[0] == reservoir.dim


def test_Layer():
    """Test Layer (i.e., no layer case)."""

    mode, distribution = 'no_layer', 'uniform'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    reservoir = ESN.layer
    weights = vars(reservoir).get('layer_w', None)
    assert weights is None


def test_DeepLayer():
    """Test DeepLayer."""

    mode, distributions = ['recurrent_layer', 'recurrent_layer'], ['uniform', 'gaussian']
    ESN = instantiate_esn(nb_layers=2, mode=mode, distribution=distributions, deep=True)
    layers = ESN.layer.layers
    is_uniform(layers[0])
    is_gaussian(layers[1])


def test_warm_up(create_dataset):
    """Test warm_up method."""

    mode, distribution = 'recurrent_layer', 'uniform'
    ESN = instantiate_esn(mode=mode, distribution=distribution)
    initial_state = ESN.layer.initial_state
    dataset_d, _ = create_dataset
    warm_up(ESN, dataset_d)
    warm_state = ESN.layer.initial_state
    assert (initial_state != warm_state).any()
