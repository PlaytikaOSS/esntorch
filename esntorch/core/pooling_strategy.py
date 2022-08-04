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


class Pooling:
    """
    Implements various pooling strategies - or pooling layers - for merging successive ESN states.

    Parameters
    ----------
    pooling_strategy : `None` or `str`
        The possible pooling strategies are:
        None, 'first', 'last', mean', 'weighted', 'lexicon_weighted'.
        None: collects all the ESN states (no pooling).
        'first': takes the first ESN state as the merged state.
        'last': takes the last ESN state as the merged state.
        'mean': takes the mean of ESN states as the merged state.
        'weighted': takes a weighted mean of ESN states as the merged state.
        'lexicon_weighted': takes a weighted mean of ESN states as the merged state.
        The weights of the words are given by a lexicon.
    weights : `None` or `torch.Tensor`
        Weights to be considered in the case of 'weighted' pooling.
        If weights is None (default), computes attention-like weights.
        The weight of each state is computed based on its norm.
        If weights != None, uses the given weights.
    lexicon : `None` or `torch.Tensor`
        Weights to be considered in the case of 'lexicon_weighted' pooling.
        If not None, 1D tensor containing the lexicon weight of each word id.
    """

    def __init__(self, pooling_strategy=None, weights=None, lexicon=None):

        self.pooling_strategy = pooling_strategy

        if weights is not None:
            self.weights = Variable(weights, requires_grad=False)
        else:
            self.weights = weights

        self.lexicon = lexicon

    def __call__(self, states, lengths, texts, additional_fts=None):
        """
        Overrides the parentheses operator.

        Parameters
        ----------
        states : `torch.Tensor`
            3D tensor containing the ESN states: [batch size x  max text length x layer dim].
        lengths : `torch.Tensor`
            1D tensor containing the lengths of each sentence in the batch: [batch size].
        texts: `transformers.tokenization_utils_base.BatchEncoding`
            Batch of token ids.
        additional_fts : `None` or `torch.Tensor`
            2D tensor containing new features (e.g. tf-idf) to be concatenated to each merged state
            [batch size x dim + additional_fts].

        Returns
        -------
        merged_states : `torch.Tensor`
            2D tensor containing merged ESN states [batch size x layer dim].
        """

        merged_states = self.merge_batch(states, lengths, texts, self.pooling_strategy, self.weights, additional_fts)

        return merged_states

    def merge_batch(self, states, lengths, texts, pooling_strategy=None, weights=None, additional_fts=None):
        """
        Implements different pooling strategies: None, 'first', 'last', 'mean', 'weighted'.

        Parameters
        ----------
        states : `torch.Tensor`
            3D tensor containing a batch of ESN states: [batch size x  max text length x layer dim].
        lengths : `torch.Tensor`
            1D tensor containing the lengths of each sentence in the batch: [batch size].
        texts: `transformers.tokenization_utils_base.BatchEncoding`
            Batch of token ids.
        pooling_strategy : `None` or `str`
            Possible values are: None, 'first', 'last', 'mean', 'weighted', 'lexicon_weighted'.
        weights : `None` or `torch.Tensor`
            2D tensor containing the weights for each state [batch size x max text length].
        additional_fts : `None` or `torch.Tensor`
            2D tensor containing new features (e.g. tf-idf) to be concatenated to each merged state
            [batch size x dim + additional_fts].

        Returns
        -------
        merged_states : `torch.Tensor`
            If pooling_strategy is not None, 2D tensor containing the merged states [batch size x layer dim].
            If pooling_strategy is None, 2D tensor containing all the states [Sum_i len(sentence_i) x layer dim].
        """

        # None: takes all (non-null) ESN states
        if pooling_strategy is None:
            merged_states = torch.cat([states[i, :j + 1, :] for i, j in enumerate(lengths - 1)], dim=0)

        # 'first': takes the first ESN state
        elif pooling_strategy == "first":
            merged_states = states[:, 0, :]

        # 'last': takes the last ESN state
        elif pooling_strategy == "last":
            merged_states = torch.stack([states[i, j, :] for i, j in enumerate(lengths - 1)], dim=0)

        # 'mean': takes the mean of ESN states
        elif pooling_strategy == "mean":
            lengths = lengths.expand(states.size()[2], -1).transpose(0, 1)
            merged_states = torch.div(torch.sum(states, dim=1), lengths)

        # 'weighted': takes a weighted mean of ESN states
        elif pooling_strategy == "weighted":
            if weights is None:
                # weights are the states' norms
                weights = Variable(torch.norm(states, dim=2), requires_grad=False)
                normalization = lengths.expand(states.size()[1], -1).double().transpose(0, 1)
                weights = torch.div(weights, normalization).type(torch.float32)

            try:
                assert tuple(weights.size())[0] == tuple(states.size())[0]
            except Exception as e:
                print('Dimensions of states and weights do not match...')
                print(e)

            merged_states = states * weights.unsqueeze(2).expand(states.size())
            merged_states = merged_states.sum(dim=1)

        # 'lexicon_weighted': takes a weighted mean of ESN states, where the weights are given by a lexicon.
        elif pooling_strategy == "lexicon_weighted":
            if self.lexicon is None:
                raise Exception('With the lexicon_weighted pooling strategy, you should pass a lexicon tensor!')

            weights = self.lexicon[texts]

            # option 0: comment options 1 and 2
            merged_states = (weights.transpose(0, 1).unsqueeze(2) * states).sum(dim=1)
            # option 1
            # merged_states /= weights.sum(dim=0)[:, None]
            # option 2
            # merged_states /= lengths.unsqueeze(1).repeat(1, states.size()[2])

        # 'mean_and_additional_fts': takes the mean of ESN states concatenated to the new features
        elif pooling_strategy == "mean_and_additional_fts":
            lengths = lengths.expand(states.size()[2], -1).transpose(0, 1)
            merged_states = torch.div(torch.sum(states, dim=1), lengths)
            if additional_fts is not None:
                merged_states = torch.cat([merged_states, additional_fts], dim=1)

        else:
            raise Exception('Unknown pooling strategy used')

        return merged_states
