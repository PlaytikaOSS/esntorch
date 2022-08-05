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
    size : `tuple`
        Size of the generated tensor.
    sparsity : `float`
        Sparsity of the generated tensor (None by default).
        Corresponds to the percentage of tensor values set to zero.
    scaling : `float`
        Scaling of the generated tensor (1.0 by default).
        Corresponds to the bounds between which the tensor values are generated.
    spectral_radius : `float`
        Spectral radius of the generated tensor (None by default).
        The spectral radius is the absolute value of the largest eigenvalue.
        The generated tensor is rescaled to have the given spectral radius.
    dtype : `torch.float32`
        Type of the generated tensor (can be changed).

    Returns
    -------
    tensor_out : `torch.Tensor`
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
    size : `tuple`
        Size of the generated tensor.
    sparsity : `float`
        Sparsity of the generated tensor (None by default).
        Corresponds to the percentage of tensor values set to zero.
    mean : `float`
        Mean of the Gaussian distribution (0.0 by default).
        Note that, after the sparsity mask has been applied and the tensor been rescaled
        according to some spectral radius, the mean of the elements will be changed.
    std : `float`
        Standard deviation of the Gaussian distribution (1.0 by default).
        Note that, after the sparsity mask has been applied and the tensor been rescaled
        according to some spectral radius, the std of the elements will be changed.
    spectral_radius : `float`
        Spectral radius of the generated tensor (None by default).
        The spectral radius is the absolute value of the largest eigenvalue.
        The generated tensor is rescaled to have the given spectral radius.
    dtype : `torch.float32`
        Type of the generated tensor (can be changed).

    Returns
    -------
    tensor_out : `torch.Tensor`
        Torch tensor generated from a Gaussian distribution.
        The tensor has the required sparsity, modified mean, modified std and spectral radius.
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


def adjust_spectral_radius(tensor_2d, spectral_radius):
    """
    Rescales a 2D tensor to have a given spectral radius.
    Converts the tensor into numpy, rescales it, and convert back into torch.
    Not optimal, but eigenvalues computation seems unstable in PyTorch.

    Parameters
    ----------
    tensor_2d : `torch.Tensor`
        2D tensor to be rescaled.
    spectral_radius : `float`
        Spectral radius obtained after rescaling.

    Returns
    -------
    tensor_2D : `torch.Tensor`
        Rescaled torch tensor with given spectral radius.
    """

    tensor_2d = tensor_2d.numpy()
    sp = np.max(np.abs(np.linalg.eigvals(tensor_2d)))
    tensor_2d = tensor_2d * (spectral_radius / sp)
    tensor_2d = torch.from_numpy(tensor_2d)

    return tensor_2d


def duplicate_labels(labels, lengths):
    """
    Duplicates labels tensor according to the tensor lengths.
    More specifically, if labels = [l1, l2, l3] and lengths = [n1, n2, n3],
    then

    the function returns [l1,...(n1 times)...,l1, l2,...(n2 times)...,l2, l3,...(n3 times)...,l3].

    Parameters
    ----------
    labels : `torch.Tensor`
        1D tensor of labels

        .
    lengths : `torch.Tensor`
        1D tensor of lengths.

    Returns
    -------
    labels_duplicated : `torch.Tensor`
        1D tensor of duplicated labels.
    """

    # For each i, duplicate labels[i] lengths[i] times, and concatenate all those.
    labels_duplicated = torch.cat([labels.view(-1)[i].repeat(lengths[i]).view(-1) for i in range(len(lengths))], dim=0)

    return labels_duplicated
