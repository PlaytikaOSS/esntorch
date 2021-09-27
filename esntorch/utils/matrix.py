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

import numpy as np
import torch


def generate_uniform_matrix(size, sparsity=None, scaling=1.0, spectral_radius=None, dtype=torch.float32):
    """
    Generates a 2D tensor from a uniform distribution.

    Parameters
    ----------
    size : tuple
        Size of the generated tensor
    sparsity : float
        Sparsity of the generated tensor (None by default).
        Corresponds to the percentage of tensor values that are set to zero.
    scaling : float
        Scaling of the generated tensor (1.0 by default).
        Corresponds to the bounds between which the tensor values are generated.
    spectral_radius : float
        Spectral radius of the generated tensor (None by default).
        The spectral radius is the absolute value of the largest eigenvalue.
        The generated tensor is rescaled to have the given spectral radius.
    dtype : torch.float32
        Type of the generated tensor (can be changed)

    Returns
    -------
    tensor_out : torch.Tensor
        Torch tensor generated from a uniform distribution.
        The tensor has the required sparsity, scaling and spectral radius.
    """

    tensor_out = torch.FloatTensor(size[0], size[1]).uniform_(-scaling, scaling)

    if sparsity is not None:
        mask = torch.zeros(size, dtype=dtype)
        mask.bernoulli_(p=(1 - sparsity))
        tensor_out *= mask

    if spectral_radius is not None:
        tensor_out = adjust_spectral_radius(tensor_out, spectral_radius)

    return tensor_out


def generate_gaussian_matrix(size, sparsity=None, mean=0.0, std=1.0, spectral_radius=None, dtype=torch.float32):
    """
    Generates a 2D tensor from a Gaussian distribution.

    Parameters
    ----------
    size : tuple
        Size of the generated tensor
    sparsity : float
        Sparsity of the generated tensor (None by default).
        Corresponds to the percentage of tensor values that are set to zero.
    mean : float
        Mean of the Gaussian distribution (0.0 by default).
    std : float
        Standard deviation of the Gaussian distribution (1.0 by default).
    spectral_radius : float
        Spectral radius of the generated tensor (None by default).
        The spectral radius is the absolute value of the largest eigenvalue.
        The generated tensor is rescaled to have the given spectral radius.
    dtype : torch.float32
        Type of the generated tensor (can be changed)

    Returns
    -------
    tensor_out : torch.Tensor
        Torch tensor generated from a Gaussian distribution.
        The tensor has the required sparsity, mean, std and spectral radius.
    """

    tensor_out = torch.empty(size, dtype=dtype)
    tensor_out = tensor_out.normal_(mean=mean, std=std)

    if sparsity is not None:
        mask = torch.zeros(size, dtype=dtype)
        mask.bernoulli_(p=(1 - sparsity))
        tensor_out *= mask

    if spectral_radius is not None:
        tensor_out = adjust_spectral_radius(tensor_out, spectral_radius)

    return tensor_out


def adjust_spectral_radius(tensor_2D, spectral_radius):
    """
    Rescales a 2D tensor to have a given spectral radius.
    Converts the tensor into numpy, rescales it and convert back into torch.
    Not optimal, but eigenvalues computation seems unstable in PyTorch.

    Parameters
    ----------
    tensor_2D : torch.Tensor
        2D tensor to be rescaled.
    spectral_radius : float
        Spectral radius obtained after rescaling.

    Returns
    -------
    tensor_2D : torch.Tensor
        Rescaled torch tensor that has the given spectral radius
    """

    tensor_2D = tensor_2D.numpy()
    sp = np.max(np.abs(np.linalg.eigvals(tensor_2D)))
    tensor_2D = tensor_2D * (spectral_radius / sp)
    tensor_2D = torch.from_numpy(tensor_2D)

    return tensor_2D


def get_row_index(tensor_2D, val=-1.0):
    """
    Gets the index i of the first row of tensor t such that t[i, :] = [val,...,val]

    Parameters
    ----------
    tensor_2D : torch.Tensor
        2D tensor.
    val :float
        Value characterizing the constant row to be detected.

    Returns
    -------
    index : int
        Index of the first row of t with constant value val.
    """

    index = tensor_2D.shape[0] - 1

    while index > 0:
        if (tensor_2D[index, 0] - val).abs() > 1e-6:
            break
        index -= 1

    return index


def duplicate_labels(labels, lengths):
    """
    Duplicates labels
    E.g.: labels = [l1, l2, l3], lengths = [n1, n2, n3]
    -> [l1,...(n1 times)...,l1, l2,...(n2 times)...,l2, l3,...(n3 times)...,l3].

    Parameters
    ----------
    labels: torch.Tensor
        1D tensor of labels
    lengths: torch.Tensor
        1D tensor of labels

    Returns
    -------
    labels_duplicated: torch.Tensor
        1D tensor of labels
    """

    # for each i, duplicate labels[i] lengths[i] times, and concatenate all those.
    labels_duplicated = torch.cat([labels.view(-1)[i].repeat(lengths[i]).view(-1) for i in range(len(lengths))], dim=0)

    return labels_duplicated


def crazyexp(x):
    """
    Modified exponential function: crazyexp(x) = exp(z) if x > 0; crazyexp(x) = 0 otherwise.
    Can be broadcasted to tensors of any dimensions.

    Parameters
    ----------
    x : float, torch.Tensor
        input of crazyexp(...)

    Returns
    -------
    x: float, torch.Tensor)
        output of crazyexp(...)
    """

    x = torch.exp(x) * (x != 0)

    return x


def crazysoftmax(vec, dim=-1):
    """
    Computes the softmax of vector, but only considering non-zero elements.
    Can be broadcasted to tensors of higher dimensions.

    Parameters
    ----------
    vec : torch.Tensor
        Vector to be softmaxed
    dim : int
        Dimension of along which the sum is performed in the process (-1 by default)

    Returns
    -------
    vec : torch.Tensor
        Softmaxed vector
    """

    vec = crazyexp(vec)/crazyexp(vec).sum(dim=dim).unsqueeze(dim)

    return vec
