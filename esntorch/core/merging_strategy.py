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
from torch.autograd import Variable
# from src.utils.matrix import crazysoftmax


class MergingStrategy:
    """
    Implements various merging strategies for grouping successive ESN states.

    Parameters
    ----------
    merging_strategy : None, str
        The possible merging strategies are: None, 'first', 'last', mean', 'weighted'.
        None: collect all ESN states (no merging).
        'first': takes the first ESN state.
        'last': takes the last ESN state.
        'mean': takes the mean of ESN states.
        'weighted': takes a weighted mean of ESN states;.
    weights : torch.Tensor
        Weights to be considered for the weighted mean merging strategy.
        If weights == None (default), computes attention-like weights.
        If weights != None, uses the given weights.
    """

    # Constructor
    def __init__(self, merging_strategy=None, weights=None, lexicon=None):
        """
        Parameters
        ----------
        merging_strategy : None, str
        weights : None, torch.Tensor
        lexicon: None, 1D tensor containing the lexicon weights at each index in the vocabulary
        """

        self.merging_strategy = merging_strategy

        if weights is not None:
            self.weights = Variable(weights, requires_grad=False)
        else:
            self.weights = weights

        self.lexicon = lexicon

    # Executor (overloads the parenthesis operator)
    def __call__(self, states, lengths, texts):
        """
        Parameters
        ----------
        states : torch.Tensor
            3D tensor containing the ESN states: (batch size x  max text length x reservoir dim).
        lengths : torch.Tensor
            1D tensor of containing text lengths: (batch size).
        texts: torch.Tensor
            2D tensor containing word indices of the texts in the batch.

        Returns
        -------
        merged_states : torch.Tensor
            2D tensor containing merged ESN states (batch size x reservoir dim).
        """

        merged_states = self.merge_batch(states, lengths, texts, self.merging_strategy, self.weights)

        return merged_states

    def merge_batch(self, states_batch, lengths, texts, merging_strategy=None, weights=None):
        """
        Implements the different merging strategies: None, 'first', 'last', 'mean', 'weighted'.

        Parameters
        ----------
        states_batch : torch.Tensor
             3D tensor containing the ESN states: (batch size x max text length x reservoir dim).
        lengths : torch.Tensor
            1D tensor containing text lengths: (batch size).
        texts: torch.Tensor
            2D tensor containing word indices of the texts in the batch.
        merging_strategy : None, str
            None, 'first', 'last', 'mean', 'weighted'.
        weights : None, torch.Tensor
            2D tensor containing the weights for each state: (batch size x max text length).

        Returns
        -------
        merged_states : torch.Tensor
            If merging_strategy is not None, 2D tensor containing the merged states: (batch size x reservoir dim).
            If merging_strategy is None, 2D tensor containing all states: (Sum_i len(state_i) x reservoir dim).
        """

        # None: takes all (non-null) ESN states
        if merging_strategy is None:
            merged_states = torch.cat([states_batch[i, :j+1, :] for i, j in enumerate(lengths - 1)], dim=0)

        # 'first': takes the first ESN state
        elif merging_strategy == "first":
            merged_states = states_batch[:, 0, :]

        # 'last': takes the last ESN state
        elif merging_strategy == "last":
            merged_states = torch.stack([states_batch[i, j, :] for i, j in enumerate(lengths - 1)], dim=0)

        # 'mean': takes the mean of ESN states
        elif merging_strategy == "mean":
            lengths = lengths.expand(states_batch.size()[2], -1).transpose(0, 1)
            merged_states = torch.div(torch.sum(states_batch, dim=1), lengths)

        # 'weighted': takes a weighted mean of ESN states
        elif merging_strategy == "weighted":
            if weights is None:
                # weights are the states' norms
                weights = Variable(torch.norm(states_batch, dim=2), requires_grad=False)
                normalization = lengths.expand(states_batch.size()[1], -1).double().transpose(0, 1)
                # old normalization (don't remember why sqrt(...))
                # normalization = torch.sqrt(lengths.expand(states_batch.size()[1], -1).double()).transpose(0, 1)
                weights = torch.div(weights, normalization).type(torch.float32)
                # uncomment (and check) the following lines to apply "softmax-based" weighted mean
                # apply attention-based weighted average
                # weights *= 1/np.sqrt(states_batch.size()[2])
                # weights = torch.nn.Softmax(dim=1)(weights)
                # weights = crazysoftmax(weights, dim=1)

            try:
                assert tuple(weights.size())[0] == tuple(states_batch.size())[0]
            except Exception as e:
                print('Dimensions of states and weights do not match...')
                print(e)

            merged_states = states_batch * weights.unsqueeze(2).expand(states_batch.size())
            merged_states = merged_states.sum(dim=1)

        elif merging_strategy == "lexicon_weighted":
            if self.lexicon is None:
                raise Exception('With the lexicon_weighted merging strategy, you should pass a lexicon tensor!')

            weights = self.lexicon[texts]

            # option 0: comment options 1 and 2
            merged_states = (weights.transpose(0, 1).unsqueeze(2)*states_batch).sum(dim=1)
            # option 1
            merged_states /= weights.sum(dim=0)[:, None]
            # option 2
            # merged_states /= lengths.unsqueeze(1).repeat(1, states_batch.size()[2])

        else:
            raise Exception('Unknown merging strategy used')

        return merged_states
