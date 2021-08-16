#   Copyright 2021 Playtika Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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

        # no reservoir in this case
        self.reservoir = res.NoneReservoir(embedding_weights=embedding_weights, seed=seed, device=device)
        self.device = self.device
