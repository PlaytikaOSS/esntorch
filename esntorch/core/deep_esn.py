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

import esntorch.core.deep_reservoir as deep_res
import esntorch.core.merging_strategy as ms
import esntorch.core.esn as esn


class DeepEchoStateNetwork(esn.EchoStateNetwork):
    """
    Implements the Deep Echo State Network (BS) per se.
    An Deep ESN consists of the combination of many reservoir, a merging strategy and a learning algorithm.

    Parameters
    ----------
    nb_reservoirs : int
        Number of reservoirs composing the deep reservoir.
    embedding_weights : torch.Tensor
        Embedding matrix for the *first* reservoir layer only (a priori).
    distributions : list of str
        List of distributions ('uniform' or 'gaussian' of the reservoirs)
    reservoir_dims : list of int
        List of reservoir dimensions.
    bias_scalings : list of float
        List of bias scaling values of the reservoirs: bounds used for the bias random generation.
    input_scalings : list of float
        List of input scaling values applied only in the case of uniform reservoirs, and ignored otherwise:
        bounds used for the input weights random generation.
    means : list of float
        List of means applied only in the case of Gaussian reservoirs, and ignored otherwise:
        Mean for Gaussian generation of reservoirs weights.
    stds : list of float
        List of standard deviations applied only in the case of Gaussian reservoirs, and ignored otherwise:
        Standard deviations for Gaussian generation of reservoirs weights.
    sparsities : list of float
        List of sparsity values of the reservoirs (between 0 and 1)
    spectral_radii : list of float
        Spectral radii of the reservoirs' weights.
        Should theoretically be below 1, but slightly above 1 works in practice.
    leaking_rates : list of float
        Leaking rate of teh reservoir (between 0 and 1).
        Determines the amount of last state and current input involved in the current state updating.
    activation_functions : list of builtin_function_or_method
        Activation function of the reservoir cells (tanh by default).
    seeds : list of int
        Random seeds.
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
                 nb_reservoirs=1,
                 embedding_weights=None,
                 distributions=None,
                 reservoir_dims=None,
                 bias_scalings=None,
                 input_scalings=None,
                 means=None,
                 stds=None,
                 sparsities=None,
                 spectral_radii=None,
                 leaking_rates=None,
                 activation_functions=None,
                 seeds=None,
                 learning_algo=None,
                 criterion=None,
                 optimizer=None,
                 merging_strategy=None,
                 bidirectional=False,
                 lexicon=None,
                 ):

        super(DeepEchoStateNetwork, self).__init__()

        # attribute should be called "reservoir" in order to properly inherit from methods of EchoStateNetwork
        self.reservoir = deep_res.DeepReservoir(nb_reservoirs=nb_reservoirs,
                                                embedding_weights=embedding_weights,
                                                distributions=distributions,
                                                reservoir_dims=reservoir_dims,
                                                bias_scalings=bias_scalings,
                                                input_scalings=input_scalings,
                                                means=means,
                                                stds=stds,
                                                sparsities=sparsities,
                                                spectral_radii=spectral_radii,
                                                leaking_rates=leaking_rates,
                                                activation_functions=activation_functions,
                                                seeds=seeds)

        self.merging_strategy = ms.MergingStrategy(merging_strategy, lexicon=lexicon)

        self.learning_algo = learning_algo
        self.criterion = criterion
        self.optimizer = optimizer
        self.bidirectional = bidirectional
