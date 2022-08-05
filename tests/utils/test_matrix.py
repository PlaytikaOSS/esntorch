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
# pytest tests/utils/test_matrix.py
# To launch all tests inside /esntorch/utils/ with line coverage, run the following command:
# pytest --cov tests/utils/
# *** END INSTRUCTIONS ***


import esntorch.utils.matrix as mt
import torch
import pytest


def test_generate_uniform_matrix():
    """Tests the generate_uniform_matrix function."""

    params_d = {'size': (50, 50),
                'sparsity': 0.95,
                'scaling': 1.2,
                'spectral_radius': 0.9,
                'dtype': torch.float32
                }

    x = mt.generate_uniform_matrix(**params_d)

    # Checks tensor size
    assert x.shape == params_d['size']
    # Checks tensor sparsity
    sparsity = torch.sum(x == 0) / (x.shape[0] * x.shape[1])
    assert params_d['sparsity'] == pytest.approx(sparsity, 0.05)
    # Checks tensor scaling
    assert x.min() >= -params_d['scaling']
    assert x.max() <= params_d['scaling']
    # Checks tensor spectral radius
    eigenvalues = torch.eig(x)[0]
    rho = max([torch.norm(lamda) for lamda in eigenvalues])
    assert rho == pytest.approx(params_d['spectral_radius'], 0.05)
    # Checks tensor dtype
    assert x.dtype == params_d['dtype']


def test_generate_gaussian_matrix():
    """Tests the generate_gaussian_matrix function."""

    params_d = {'size': (100, 100),
                'sparsity': 0.95,
                'mean': 1.7,
                'std': 0.6,
                'spectral_radius': 0.9,
                'dtype': torch.float32
                }

    x = mt.generate_gaussian_matrix(**params_d)

    # Checks tensor size
    assert x.shape == params_d['size']
    # Checks tensor sparsity
    sparsity = torch.sum(x == 0) / (x.shape[0] * x.shape[1])
    assert params_d['sparsity'] == pytest.approx(sparsity, 0.05)
    # Checks tensor spectral radius
    eigenvalues = torch.eig(x)[0]
    rho = max([torch.norm(lamda) for lamda in eigenvalues])
    assert rho == pytest.approx(params_d['spectral_radius'], 0.05)
    # Checks tensor dtype
    assert x.dtype == params_d['dtype']

    # When sparsity is None and spectral_radius is None, the mean and std can be tested (since no rescaling).
    params_d = {'size': (1000, 1000),
                'sparsity': None,
                'mean': 1.7,
                'std': 0.6,
                'spectral_radius': None,
                'dtype': torch.float32
                }

    x = mt.generate_gaussian_matrix(**params_d)
    # Checks tensor mean
    print('MEAN', x.mean())
    assert x.mean() == pytest.approx(params_d['mean'], 0.05)
    # Checks tensor std
    assert x.std() == pytest.approx(params_d['std'], 0.05)


def test_adjust_spectral_radius():
    """Tests the adjust_spectral_radius function."""

    x = torch.rand(size=(30, 30))
    rho_x = 0.94
    x = mt.adjust_spectral_radius(x, rho_x)

    # Checks tensor spectral radius
    eigenvalues = torch.eig(x)[0]
    rho = max([torch.norm(lamda) for lamda in eigenvalues])
    assert rho == pytest.approx(rho_x, 0.01)


def test_duplicate_labels():
    """Tests the duplicate_labels function."""

    labels = torch.randint(low=0, high=5, size=(7,))
    lengths = torch.randint(low=10, high=20, size=(7,))
    x = mt.duplicate_labels(labels, lengths)

    for i, (label, n) in enumerate(zip(labels, lengths)):
        y = x[:n]
        assert torch.all(y == torch.tensor([label] * n))
        x = x[n:]
