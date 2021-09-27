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

import random
import torch
import torch.nn as nn
import esntorch.core.reservoir as res


class DeepReservoir(nn.Module):
    """
    Implements a deep reservoir, to be used in the context of a deep echo state network (DeepESN).
    Parameters are self-explanatory.

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
                 seeds=None):

        super(DeepReservoir, self).__init__()

        # Initialize embeddings and set pre-trained weights
        vocab_size, embed_dim = embedding_weights.size()
        self.embedding = nn.Embedding(vocab_size, embed_dim).requires_grad_(False)
        self.embedding.weight.data.copy_(embedding_weights)

        # Assign a random embedding to the <unk> token
        self.embedding.weight.data[0] = torch.rand(embed_dim)

        # Number of reservoirs
        self.nb_reservoirs = nb_reservoirs

        # Distribution of reservoirs
        self.distributions = ['uniform']*self.nb_reservoirs if distributions is None else distributions

        # Dimensions of reservoirs
        self.reservoir_dims = reservoir_dims

        # Dimensions of inputs (inferred from reservoir dims)
        self.input_dims = [embed_dim] + [d for d in self.reservoir_dims][:-1]

        # Bias scalings
        self.bias_scalings = [1.0]*self.nb_reservoirs if bias_scalings is None else bias_scalings

        # Input scalings
        self.input_scalings = [1.0]*self.nb_reservoirs if input_scalings is None else input_scalings

        # Means
        self.means = [0.0]*self.nb_reservoirs if means is None else means

        # Stds
        self.stds = [1.0]*self.nb_reservoirs if stds is None else stds

        # Sparsities
        self.sparsities = [0.99]*self.nb_reservoirs if sparsities is None else sparsities

        # Spectral radii
        self.spectral_radii = [0.9]*self.nb_reservoirs if spectral_radii is None else spectral_radii

        # Leaking rates
        self.leaking_rates = [0.5]*self.nb_reservoirs if leaking_rates is None else leaking_rates

        # Activation functions
        self.activation_functions = ['tanh']*self.nb_reservoirs if activation_functions is None \
            else activation_functions

        # Random seeds
        self.seeds = random.sample(range(0, 1000), self.nb_reservoirs) if seeds is None else seeds

        # Generate all reservoirs composing the deep reservoir
        self.reservoirs = []

        # Check definition of reservoirs: lists of attributes should have same lengths
        attributes_l = [self.distributions, self.input_dims, self.reservoir_dims,
                        self.bias_scalings, self.input_scalings, self.means, self.stds,
                        self.sparsities, self.spectral_radii, self.leaking_rates,
                        self.activation_functions, self.seeds]

        it = iter(attributes_l)

        if not all(len(attr_l) == self.nb_reservoirs for attr_l in it):
            raise ValueError('Reservoirs ill-defined: lists of attributes should have same length!')

        # Instantiate reservoirs according to attributes
        indices = list(range(self.nb_reservoirs))

        for i in indices:

            if self.distributions[i] == 'uniform':
                self.reservoirs.append(res.UniformReservoir(embedding_weights=[embedding_weights, None][i != 0],
                                                            input_dim=self.input_dims[i],
                                                            input_scaling=self.input_scalings[i],
                                                            reservoir_dim=self.reservoir_dims[i],
                                                            bias_scaling=self.bias_scalings[i],
                                                            sparsity=self.sparsities[i],
                                                            spectral_radius=self.spectral_radii[i],
                                                            leaking_rate=self.leaking_rates[i],
                                                            activation_function=self.activation_functions[i],
                                                            seed=self.seeds[i]))

            elif self.distributions[i] == 'gaussian':
                self.reservoirs.append(res.GaussianReservoir(embedding_weights=[embedding_weights, None][i != 0],
                                                             input_dim=self.input_dims[i],
                                                             reservoir_dim=self.reservoir_dims[i],
                                                             bias_scaling=self.bias_scalings[i],
                                                             sparsity=self.sparsities[i],
                                                             spectral_radius=self.spectral_radii[i],
                                                             leaking_rate=self.leaking_rates[i],
                                                             mean=self.means[i],
                                                             std=self.stds[i],
                                                             activation_function=self.activation_functions[i],
                                                             seed=self.seeds[i]))

    def forward(self, inputs):
        """
        Implements forwards pass, i.e., processing of a batch of input texts by the successive reservoirs.
        This method uses the forward method of each reservoir

        Parameters
        ----------
        inputs : torch.Tensor
            2D input tensor (max_length x batch_size).
            A batch of input texts is a 2D tensor.
            Each tensor column represents a text - given as the sequence of its word indices.

        Returns
        -------
        concatenated_states, lengths : torch.Tensor, torch.Tensor
            concatenated_states : 3D tensor (batch_size x max_length x reservoir_dim).
            Reservoir states obtained after processing the batch of inputs into the successive reservoirs.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        """

        # Initial input: batch of texts
        current_inputs = inputs

        # Forward pass through all reservoirs
        states_l = []

        for reservoir in self.reservoirs:

            states, lengths = reservoir.forward(current_inputs)
            states_l.append(states)
            current_inputs = states

        concatenated_states = torch.cat(states_l, dim=2)

        return concatenated_states, lengths

    def reverse_forward(self, batch_tokens, lengths):
        """
        Returns the concatenated states obtained after processing the batch of texts in the successive reservoirs
        in the inverse order.

        It applies the forward on the reversed tokens in a batch
        (it will not inverse the padding tokens).

        The original lengths of the padded token sentences must be supplied.

        Parameters
        ----------
        batch_tokens : torch.Tensor
            2D tensor of the batch tokens.
        lengths : torch.Tensor
            1D tensor, the token sentences true lengths.

        Returns
        -------
        reversed_concatenated_states : torch.Tensor
            3D tensor of the batch of states of the reversed input, with the padded states in the correct place.
        """

        # first reverse the batch tokens, without affecting the padding tokens
        reversed_batch = batch_tokens.clone()

        for i, l in enumerate(lengths):
            reversed_batch[:l, i] = torch.flip(reversed_batch[:l, i], [0])

        # make the reversed batch go through the reservoir
        reversed_concatenated_states, _ = self.forward(reversed_batch)

        return reversed_concatenated_states, lengths

    def warm_up(self, warm_up_sequence):
        """
        Performs forward pass of an input sequence.
        For each reservoir, set its last reservoir state as its new initial state.
        This method uses the warm_up method of each reservoir

        Parameters
        ----------
        warm_up_sequence : torch.Tensor
            1D tensor: word indices of the warm up sentence.
        """

        # Add a first dimension to sequence to match 2D input batch format
        warm_up_sequence = warm_up_sequence.unsqueeze(1)

        # Process text into the sequence of reservoirs
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)

        # set new initial states of each reservoir
        last_state, index = warm_states[-1, -1, :], 0

        for reservoir in self.reservoirs:

            dim = reservoir.reservoir_dim
            reservoir.initial_state = last_state[index: index + dim]
            index = index + dim
