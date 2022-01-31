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

import torch

import esntorch.core.reservoir as res
import esntorch.core.esn as esn


class Baseline(esn.EchoStateNetwork):
    """
    Implements the baseline algorithms (logistic regression (LR), deep neural net (DNN), etc.).
    Here, The baseline algorithms are simply ESNs where the reservoir step is omitted.
    In other words, a baseline consists of the combination of a merging strategy and a learning algorithm.
    This class inherits from EchoStateNetwork

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Embedding matrix.
    input_dim : int
        Input dimension.
    learning_algo : src.models.learning_algo.RidgeRegression, src.models.learning_algo.LogisticRegression
        Learning algorithm used to learn the targets from the reservoir (merged) states.
    criterion : torch.nn.modules.loss
        Criterion used to compute the loss between tagets and predictions (only if leaning_algo ≠ RidgeRegression).
    optimizer : torch.optim
        Optimizer used in the gradient descent method (only if leaning_algo ≠ RidgeRegression).
    merging_strategy : src.models.merging_strategy.MergingStrategy
        Merging strategy used to merge the sucessive reservoir states.
    bidirectional : bool
        Flag for bi-directionality.
    seed : torch._C.Generator
        Random seed.
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 learning_algo=None,
                 criterion=None,
                 optimizer=None,
                 merging_strategy=None,
                 lexicon=None,
                 bidirectional=False,
                 seed=42,
                 device=torch.device('cpu')
                 ):

        super(Baseline, self).__init__(embedding_weights=embedding_weights,
                                       input_dim=input_dim,
                                       learning_algo=learning_algo,
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       merging_strategy=merging_strategy,
                                       lexicon=lexicon,
                                       bidirectional=bidirectional,
                                       seed=seed,
                                       device=device)

        # No reservoir in this case
        self.reservoir = res.NoneReservoir(embedding_weights=embedding_weights, seed=seed, device=device)
        self.device = self.device


class CustomBaseline(esn.EchoStateNetwork):
    """
    Implements the Custom Baseline, which consists of an embedding layer (EMB)
    followed by a fully connected layer (FC) followed by a learning algorithm (LA)
    (e.g., Ridge regression).
    Note that the FC layer is not trained.
    Custom RR-Baseline = EMB + FC + LA
    This class inherits from EchoStateNetwork

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Embedding matrix.
    input_dim : int
        Input dimension.
    reservoir_dim : int
        Dimension of the hidden layer onto which the embedded inputs are projected.
    input_scaling : float
        Bounds used for the uniform random generation of the input weights.
    bias_scaling : float
        Bounds used for the uniform random generation of the bias weights.
    activation_function : str
        Activation function of the reservoir cells ('tanh' by default).
    learning_algo : Union[src.models.learning_algo.RidgeRegression, src.models.learning_algo.LogisticRegression,...]
        Learning algorithm used to learn the association between the reservoir (merged) states and the targets.
    criterion : torch.nn.modules.loss
        Criterion used to compute the loss between tagets and predictions (only if leaning_algo ≠ RidgeRegression).
    optimizer : torch.optim
        Optimizer used in the gradient descent method (only if leaning_algo ≠ RidgeRegression).
    merging_strategy : src.models.merging_strategy.MergingStrategy
        Merging strategy used to merge the sucessive reservoir states.
    bidirectional : bool
        Flag for bi-directionality.
    seed : torch._C.Generator
        Random seed.
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 reservoir_dim=None,
                 input_scaling=None,
                 bias_scaling=None,
                 activation_function='tanh',
                 learning_algo=None,
                 criterion=None,
                 optimizer=None,
                 merging_strategy=None,
                 lexicon=None,
                 bidirectional=False,
                 seed=42,
                 device=torch.device('cpu')
                 ):

        super(CustomBaseline, self).__init__(embedding_weights=embedding_weights,
                                             input_dim=input_dim,
                                             reservoir_dim=reservoir_dim,
                                             input_scaling=input_scaling,
                                             bias_scaling=bias_scaling,
                                             activation_function=activation_function,
                                             learning_algo=learning_algo,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             merging_strategy=merging_strategy,
                                             lexicon=lexicon,
                                             bidirectional=bidirectional,
                                             seed=seed,
                                             device=device)

        # Reservoir is an FC in this case
        self.reservoir = res.ReservoirFC(embedding_weights=embedding_weights,
                                         input_dim=input_dim,
                                         reservoir_dim=reservoir_dim,
                                         input_scaling=input_scaling,
                                         bias_scaling=bias_scaling,
                                         activation_function=activation_function,
                                         seed=seed, device=device)
        self.device = self.device
