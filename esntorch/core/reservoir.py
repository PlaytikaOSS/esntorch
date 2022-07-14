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
import torch.nn as nn
import esntorch.utils.matrix as mat
import esntorch.utils.embedding as emb
from torch.autograd import Variable


class Layer(nn.Module):
    """
    Implements a Layer of a network. A layer is composed of an embedding and a forward method.
    This is a base class for more complex layer, like LayerLinear or LayerRecurrent.

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Embedding matrix.
    seed : torch._C.Generator
        Random seed.
    device: torch.device
        Device gpu or cpu.
    """

    def __init__(self,
                 embedding_weights=None,
                 seed=42,
                 device=torch.device('cpu'),
                 **kwargs):

        super().__init__()

        torch.manual_seed(seed)
        self.device = device

        # Set embeddings
        # TorchText (embedding weights)
        if (embedding_weights is not None) and torch.is_tensor(embedding_weights):
            vocab_size, embed_dim = embedding_weights.size()
            self.embedding = nn.Embedding(vocab_size, embed_dim).requires_grad_(False)
            self.embedding.weight.data.copy_(embedding_weights)

            # Assign a random embedding to the <unk> token
            self.embedding.weight.data[0] = torch.rand(embed_dim)

        # HuggingFace (embedding function)
        elif isinstance(embedding_weights, str):
            self.HuggingFaceEmbedding = emb.EmbeddingModel(embedding_weights, device)
            self.embedding = lambda batch: self.HuggingFaceEmbedding.get_embedding(batch)

        # No embedding, for deep reservoirs
        else:
            self.embedding = None

    def _embed(self, batch):

        if callable(self.embedding):
            batch_size = int(batch["input_ids"].shape[0])
            # Ignore the first [CLS] and the last [SEP] tokens (hence lengths - 2)
            lengths = batch["lengths"].to(self.device) - 2
            embedded_inputs = self.embedding(batch)
            embedded_inputs = embedded_inputs[1:, :, :]  # Ignore [CLS]
        else:
            # batch_size = int(batch.size()[0])
            batch_size = int(batch.shape[1])  # XXX
            # lengths = batch.sum(dim=2).shape[1] - (batch.sum(dim=2) == 0.0).sum(dim=1)
            lengths = (batch.sum(dim=2) != 0.0).sum(dim=0)  # XXX
            embedded_inputs = batch  # torch.transpose(batch, 0, 1) # XXX

        return batch_size, lengths, embedded_inputs

    def _forward(self, batch_size, lengths, embedded_inputs):
        states = embedded_inputs.transpose(0, 1)
        return states, lengths

    def forward(self, batch):
        return self._forward(*self._embed(batch))

    def reverse_forward(self, batch):
        # flip tensor
        batch_rev = batch.copy()
        batch_rev["input_ids"] = batch["input_ids"].clone()

        for i, l in enumerate(batch["lengths"]):
            batch_rev["input_ids"][i, :l] = torch.flip(batch["input_ids"][i, :l], dims=[0])

        # make the reversed batch go through the layer
        return self.forward(batch_rev)

    def warm_up(self, warm_up_sequence, return_states=False):
        """
        Performs forward pass of an input sequence and set last layer state as new initial state.

        Parameters
        ----------
        warm_up_sequence : torch.Tensor
            1D tensor: word indices of the warm up sentence.
        """

        if not callable(self.embedding):  # TorchText
            # Add a first dimension to sequence to match 2D input batch format
            warm_up_sequence = warm_up_sequence.unsqueeze(1)

        # Process text into the layer
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)
        # Remove first dimension and take last valid state
        self.initial_state = warm_states[0, warm_sentence_length - 1, :].reshape(-1)

        if return_states:
            return warm_states


class LayerLinear(Layer):
    """
    Implements the layer of an echo state network (ESN).
    The required parameters are self-explanatory.

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Embedding matrix.
    input_dim : int
        Input dimension.
    dim : int
        Reservoir dimension.
    bias_scaling : float
        Bias scaling: bounds used for the bias random generation.
    leaking_rate : float
        Leaking rate of the layer (between 0 and 1).
        Determines the amount of last state and current input involved in the current state updating.
    activation_function : str
        Activation function of the layer cells ('tanh' by default).
    seed : torch._C.Generator
        Random seed.
    device: torch.device
        Device gpu or cpu.
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 dim=None,
                 bias_scaling=None,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu'),
                 **kwargs
                 ):

        super().__init__(embedding_weights=embedding_weights, seed=seed, device=device)

        # Scalings
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        # Input and layer dim
        self.input_dim = input_dim
        self.dim = dim

        # Activation function
        if activation_function == 'tanh':
            self.activation_function = torch.tanh
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            print("Activation function unknown...")
            self.activation_function = None

        input_w = mat.generate_uniform_matrix(size=(self.dim, self.input_dim),
                                              scaling=self.input_scaling)
        input_w = Variable(input_w, requires_grad=False)
        self.register_buffer('input_w', input_w)

        # Bias
        if self.bias_scaling is not None:
            bias = mat.generate_uniform_matrix(size=(1, self.dim),
                                               scaling=self.bias_scaling).flatten()
        else:
            bias = torch.zeros(size=(1, self.dim)).flatten()
        bias = Variable(bias, requires_grad=False)
        self.register_buffer('bias', bias)

        # Initial_state
        self.register_buffer('initial_state', torch.rand(self.dim, requires_grad=False))

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Implements the processing of a batch of input texts into a linear layer \
        of same dimension as the layer.
        This is the so-called "Custom Baseline" (EMB + LINEAR_LAYER + LA).

        Parameters
        ----------
        batch_size : int
            Batch size.
        lengths : torch.Tensor
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        embedded_inputs : torch.Tensor
            embedded_inputs : 3D tensor (batch_size x max_length x dim).
            Embedded inputs is the 3D tensor obtained after embedding the batch of input tokens.
            Note that the embeddings of the tokens [CLS] and [SEP] generated by
            the Hugging Face tokenizer are discarded from the embedded_inputs tensor.

        Returns
        -------
        states, lengths : torch.Tensor, torch.Tensor
            See the method _forward_esn() for further details.
        """

        # States: left uninitialized to speed up things
        states = torch.empty(batch_size, lengths.max(), self.dim, dtype=torch.float32, device=self.device)

        # For each time step, we process all sentences in the batch concurrently
        for t in range(lengths.max()):
            # Current input (embedded word)
            u = embedded_inputs[t, :, :]

            # Input activation
            u_act = torch.mm(self.input_w, u.transpose(0, 1))

            # New layer state f(input_act + reservoir_act)
            x_new = self.activation_function(u_act + self.bias.expand(batch_size, -1).transpose(0, 1))

            # Amazing one-liner to zero any layer state after the respective sentence end
            x_new *= (lengths - t > 0)

            # Add new layer state to states
            states[:, t, :] = x_new.transpose(0, 1)

        #         # Possible loop-free implementation (to be verified)
        #         # repeat input_w and bias along the batch dimension
        #         input_w = self.input_w.repeat(batch_size, 1, 1).transpose(1,2)
        #         bias = self.bias.repeat(batch_size, sentence_dim, 1)
        #         states = self.activation_function(torch.bmm(embedded_inputs, input_w) + bias)

        return states, lengths


class LayerRecurrent(LayerLinear):
    """
    Implements the layer of an echo state network (ESN).
    The required parameters are self-explanatory.

    Parameters
    ----------
    embedding_weights : torch.Tensor
        Embedding matrix.
    input_dim : int
        Input dimension.
    reservoir_dim : int
        Reservoir dimension.
    bias_scaling : float
        Bias scaling: bounds used for the bias random generation.
    distribution: str
        Distribution used for layer weights ('gaussian' or 'uniform').
    mean: float
        Mean of distribution.
    std: float
        Standard deviation of distribution.
    sparsity : float
        Sparsity of the layer (between 0 and 1).
    spectral_radius : float
        Spectral radius of the layer weights.
        Should theoretically be below 1, but slightly above 1 works in practice.
    leaking_rate : float
        Leaking rate of teh layer (between 0 and 1).
        Determines the amount of last state and current input involved in the current state updating.
    activation_function : str
        Activation function of the layer cells ('tanh' by default).
    seed : torch._C.Generator
        Random seed.
    device: torch.device
        Device gpu or cpu.
    """

    # Constructor
    def __init__(self,
                 distribution='gaussian',
                 mean=0.0,
                 std=1.0,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 **kwargs):

        super().__init__(**kwargs)

        # Distribution, mean, std, sparsity, spectral_radius, leaking_rate
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate

        # Reservoir
        if distribution == 'gaussian':
            layer_w = mat.generate_gaussian_matrix(size=(self.dim, self.dim),
                                                   sparsity=self.sparsity,
                                                   mean=self.mean,
                                                   std=self.std,
                                                   spectral_radius=self.spectral_radius)
        elif distribution == 'uniform':
            layer_w = mat.generate_uniform_matrix(size=(self.dim, self.dim),
                                                  scaling=1.,
                                                  sparsity=self.sparsity,
                                                  spectral_radius=self.spectral_radius)
        else:
            raise ValueError('distribution not handled')

        layer_w = Variable(layer_w, requires_grad=False)
        self.register_buffer('layer_w', layer_w)

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Implements the processing of a batch of input texts into the layer.

        Parameters
        ----------
        batch : Union[transformers.tokenization_utils_base.BatchEncoding, torch.Tensor]
            In the case of a dynamic embedding, like BERT,
            batch is a BatchEncoding object output by a Hugging Face tokenizer.
            Usually, batch contains different keys, like 'attention_mask', 'input_ids', 'labels', 'lengths'...
            batch['input_ids'] is a 2D tensor (batch_size x max_length) of the form:

            ::

                tensor([[ 101, 2129, 2001,  ...,    0,    0,    0],
                        [ 101, 2054, 2003,  ...,  102,    0,    0],
                        [ 101, 2073, 2003,  ...,  102,    0,    0],
                        ...,
                        [ 101, 2054, 2001,  ..., 7064, 1029,  102],
                        [ 101, 2054, 2024,  ..., 2015, 1029,  102],
                        [ 101, 2073, 2003,  ..., 2241, 1029,  102]])

            This tensor is composed by the tokenized sentences of the batch stacked horizontally.
            In the case of a static embedding, batch is a 2D tensor (batch_size x max_length) of the form:

            ::

                tensor([[ 101, 2129, 2001,  ...,    0,    0,    0],
                        [ 101, 2054, 2003,  ...,  102,    0,    0],
                        [ 101, 2073, 2003,  ...,  102,    0,    0],
                        ...,
                        [ 101, 2054, 2001,  ..., 7064, 1029,  102],
                        [ 101, 2054, 2024,  ..., 2015, 1029,  102],
                        [ 101, 2073, 2003,  ..., 2241, 1029,  102]])

            This tensor is composed by the tokenized sentences of the batch stacked horizontally.

        Returns
        -------
        states, lengths : torch.Tensor, torch.Tensor
            states : 3D tensor (batch_size x max_length x dim).
            Reservoir states obtained after processing the batch of inputs into the layer.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        """

        # Set initial layer state
        current_reservoir_states = self.initial_state.expand(batch_size, -1).transpose(0, 1)

        # States: left uninitialized to speed up things
        states = torch.empty(batch_size, lengths.max(), self.dim, dtype=torch.float32, device=self.device)

        # For each time step, we process all sentences in the batch concurrently
        for t in range(lengths.max()):
            # Current input (embedded word)
            u = embedded_inputs[t, :, :]

            # Input activation
            u_act = torch.mm(self.input_w, u.transpose(0, 1))

            # Current layer state
            x = current_reservoir_states

            # Reservoir activation
            x_act = torch.mm(self.layer_w, x)

            # New layer state f(input_act + reservoir_act)
            x_new = self.activation_function(u_act + x_act + self.bias.expand(batch_size, -1).transpose(0, 1))

            x_new = (1 - self.leaking_rate) * x + self.leaking_rate * x_new

            # Amazing one-liner to zero any layer state after the respective sentence end
            x_new *= (lengths - t > 0)

            # Add new layer state to states
            states[:, t, :] = x_new.transpose(0, 1)

            # New layer state becomes current layer state
            current_reservoir_states = x_new

        return states, lengths


def create_layer(mode='recurrent_layer', **kwargs):
    """
    Create a recurrent, linear or empty layer depending on the given mode.
        mode : str
            Either "recurrent_layer",  "linear_layer", "no_layer".
    """
    if mode == 'recurrent_layer':
        return LayerRecurrent(**kwargs)
    elif mode == 'linear_layer':
        return LayerLinear(**kwargs)
    elif mode == 'no_layer':
        return Layer(**kwargs)
    else:
        raise ValueError('wrong mode')


class DeepLayer(Layer):
    """
    Implements a deep layer, to be used in the context of a deep echo state network (DeepESN).
    Parameters are self-explanatory.

    Parameters
    ----------
    nb_layers : int
        Number of reservoirs composing the deep layer.
    embedding_weights : torch.Tensor
        Embedding matrix for the *first* layer layer only (a priori).
    distributions : list of str
        List of distributions ('uniform' or 'gaussian' of the reservoirs)
    reservoir_dims : list of int
        List of layer dimensions.
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
        Leaking rate of teh layer (between 0 and 1).
        Determines the amount of last state and current input involved in the current state updating.
    activation_functions : list of builtin_function_or_method
        Activation function of the layer cells (tanh by default).
    seeds : list of int
        Random seeds.
    """

    # Constructor
    def __init__(self,
                 nb_layers=1,
                 embedding_weights=None,
                 seed=42,
                 device=torch.device('cpu'),
                 **kwargs
                 ):

        super().__init__(embedding_weights=embedding_weights, seed=seed, device=device)

        self.nb_layers = nb_layers

        def get_parameters(index):

            new_kwargs = {}

            for key, value in kwargs.items():
                if key == 'input_dim':
                    new_kwargs[key] = self.layers[index - 1].dim if index > 0 else value
                elif isinstance(value, list) or isinstance(value, tuple):
                    if len(value) != nb_layers:
                        raise TypeError('length of parameter does not match the number of layer')
                    else:
                        new_kwargs[key] = value[index]
                elif value is None:
                    pass
                else:
                    new_kwargs[key] = value

            return new_kwargs

        # Generate all reservoirs composing the deep layer
        self.layers = []

        for i in range(self.nb_layers):
            self.layers.append(create_layer(**get_parameters(i)))

        # XXX
        for i, layer in enumerate(self.layers):
            input_w = Variable(layer.input_w, requires_grad=False)
            self.register_buffer(f'input_{i}_w', input_w)
            layer.input_w = input_w
            # self.register_buffer(f'layer_{i}_w', layer.layer_w)
        # XXX

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Implements forwards pass, i.e., processing of a batch of input texts by the successive layers.
        This method uses the forward method of each layer

        Parameters
        ----------
        embedded_inputs : torch.Tensor
            2D input tensor (max_length x batch_size).
            A batch of input texts is a 2D tensor.
            Each tensor column represents a text - given as the sequence of its word indices.

        Returns
        -------
        concatenated_states, lengths : torch.Tensor, torch.Tensor
            concatenated_states : 3D tensor (batch_size x max_length x dim).
            Reservoir states obtained after processing the batch of inputs into the successive reservoirs.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        """

        # Initial input: batch of texts
        current_inputs = embedded_inputs

        # Forward pass through all reservoirs
        states_l = []

        for layer in self.layers:
            # states, lengths = layer.forward(current_inputs)
            states, lengths = layer._forward(batch_size, lengths, current_inputs)  # XXX
            states_l.append(states)
            # current_inputs = states
            current_inputs = states.transpose(0, 1)  # XXX

        concatenated_states = torch.cat(states_l, dim=2)

        return concatenated_states, lengths

    def warm_up(self, warm_up_sequence, return_states=False):
        """
        Performs forward pass of an input sequence.
        For each layer, set its last layer state as its new initial state.
        This method uses the warm_up method of each layer

        Parameters
        ----------
        warm_up_sequence : torch.Tensor
            1D tensor: word indices of the warm up sentence.
        """

        # Add a first dimension to sequence to match 2D input batch format
        warm_up_sequence = warm_up_sequence.unsqueeze(1)

        # Process text into the sequence of reservoirs
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)

        # set new initial states of each layer
        last_state, index = warm_states[-1, -1, :], 0

        for layer in self.layers:
            dim = layer.dim
            layer.initial_state = last_state[index: index + dim]
            index = index + dim

        if return_states:
            return warm_states
