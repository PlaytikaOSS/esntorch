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
# pytest tests/core/test_pooling_strategy.py
# To launch all tests inside /esntorch/core/ with line coverage, run the following command:
# pytest --cov tests/core/
# *** END INSTRUCTIONS ***
import torch

from esntorch.core.pooling_strategy import Pooling
import pytest


def test_init():
    """Tests the __init__ method."""

    weights = torch.randint(low=0, high=3, size=(32, 9))
    lexicon = torch.randint(low=0, high=3, size=(50,))

    m = Pooling(pooling_strategy='mean', weights=weights, lexicon=lexicon)
    assert m.pooling_strategy == 'mean'
    assert torch.all(m.weights == weights)
    assert torch.all(m.lexicon == lexicon)

    m = Pooling(pooling_strategy='mean')
    assert m.weights is None
    assert m.lexicon is None


def test_merge_batch():
    """Test the merge_batch method."""

    states = torch.rand(size=(32, 9, 300))
    lengths = torch.randint(low=1, high=9, size=(32,))
    for i, l in enumerate(lengths):
        states[i, l:, :] = 0.0
    texts = torch.randint(low=0, high=50, size=(9, 32))
    weights = torch.randint(low=0, high=3, size=(32, 9))
    lexicon = torch.randint(low=0, high=3, size=(50,))

    # Test pooling strategy None
    m = Pooling(pooling_strategy=None)
    x = m(states=states, lengths=lengths, texts=texts)
    assert x.shape == (torch.sum(lengths), states.shape[2])
    for i, l in enumerate(lengths):
        y = x[:l]
        assert torch.all(y == states[i, :l, :])
        x = x[l:]

    # Test pooling strategy 'first'
    m = Pooling(pooling_strategy='first')
    x = m(states=states, lengths=lengths, texts=texts)
    assert x.shape == (states.shape[0], states.shape[2])
    assert torch.all(x == states[:, 0, :])

    # Test pooling strategy 'last'
    m = Pooling(pooling_strategy='last')
    x = m(states=states, lengths=lengths, texts=texts)
    assert x.shape == (states.shape[0], states.shape[2])
    for i, l in enumerate(lengths):
        assert torch.all(x[i, :] == states[i, l - 1, :])

    # Test pooling strategy 'mean'
    m = Pooling(pooling_strategy='mean')
    x = m(states=states, lengths=lengths, texts=texts)
    assert x.shape == (states.shape[0], states.shape[2])
    for i, l in enumerate(lengths):
        diff = torch.all(x[i, :] == torch.mean(states[i, :l, :], axis=0))
        assert pytest.approx(diff, 0.0001)

    # Test pooling strategy 'weighted'
    m = Pooling(pooling_strategy='weighted')
    x = m(states=states, lengths=lengths, texts=texts)
    assert x.shape == (states.shape[0], states.shape[2])
    for i, l in enumerate(lengths):
        mean_state = torch.vstack([states[i, j, :] * weights[i, j] for j in range(l)])
        mean_state = torch.sum(mean_state, dim=0) / l
        diff = torch.all(x[i, :] == mean_state)
        assert pytest.approx(diff, 0.0001)

    # Test pooling strategy 'lexicon_weighted'
    m = Pooling(pooling_strategy='lexicon_weighted', lexicon=lexicon)
    x = m(states=states, lengths=lengths, texts=texts)
    weights = lexicon[texts].transpose(0, 1)
    assert x.shape == (states.shape[0], states.shape[2])
    for i, l in enumerate(lengths):
        mean_state = torch.vstack([states[i, j, :] * weights[i, j] for j in range(l)])
        mean_state = torch.sum(mean_state, dim=0) / weights.sum(dim=1)[i]
        diff = torch.all(x[i, :] == mean_state)
        assert pytest.approx(diff, 0.0001)
