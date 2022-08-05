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
    Implements a **layer** of a network.
    A general layer is composed of an **embedding** and a **reservoir** of neurons:

    LAYER = EMBEDDING + RESERVOIR.

    The layer takes a tokenized text as input, embeds it, and pass it through the reservoir
    (possibly in both directions).
    Sometimes, the embedding can be omitted (case of deep ESNs)
    and the reservoir can also be omitted (case of a baseline).
    This class a base class for more complex layers: ``LayerLinear``, ``LayerRecurrent``, ``DeepLayer``.

    Parameters
    ----------
    embedding : `None` or `str`
        Name of Hugging Face model used for embedding or None.
        The None case is used to implement deep ESNs, i.e.,
        cases in which the inputs are given by a previous reservoir instead of an embedding.
    input_dim : `int`
        Dimension of the inputs.
        If a Hugging Face model is given as an embedding,
        then input_dim is automatically set to the dimension of this model.
        Otherwise, input_dim needs to be specified, cf. case of deep ESNs (DeepLayer).
    seed : `int`
        Random seed.
    device : `torch.device`
        The device to be used: cpu or gpu.
    **kwargs : optional
        Other keyword arguments.
    """

    def __init__(self,
                 embedding=None,
                 input_dim=None,
                 seed=42,
                 device=torch.device('cpu'),
                 **kwargs):

        super().__init__()

        torch.manual_seed(seed)
        self.device = device

        # Hugging Face model as embedding
        if isinstance(embedding, str):
            self.HuggingFaceEmbedding = emb.EmbeddingModel(embedding, device)
            self.embedding = lambda batch: self.HuggingFaceEmbedding.get_embedding(batch)
            self.input_dim = self.HuggingFaceEmbedding.model.config.hidden_size

        # No embedding: case of deep ESNs
        elif embedding is None:
            self.embedding = None
            self.input_dim = input_dim

        self.dim = 0

    def _embed(self, batch):
        """
        Embeds an input batch.
        If the input batch is a consists of tokenized texts, then it will be embedded into a 3D tensor.
        If the input batch is a consists of reservoir states (case of deep ESNs), then no embedding is applied.

        Parameters
        ----------
        batch : `transformers.tokenization_utils_base.BatchEncoding` or `torch.Tensor`
            Input batch to be (potentially) embedded.

        Returns
        -------
        `tuple` [`int`, `torch.Tensor`, `torch.Tensor`]
            (batch_size, lengths, embedded_inputs):
            (i) batch_size is the batch size.
            (ii) lengths is a 1D tensor [batch size] containing the lengths of all sentences in the batch.
            (iii) embedded_inputs is a 3D tensor [max length x batch size x embedding dim]
            containing the embedded inputs.
        """

        if callable(self.embedding):
            batch_size = int(batch["input_ids"].shape[0])
            # Ignore the first [CLS] and the last [SEP] tokens (hence lengths - 2)
            lengths = batch["lengths"].to(self.device) - 2
            embedded_inputs = self.embedding(batch)
            embedded_inputs = embedded_inputs[1:, :, :].to(self.device)  # Ignore [CLS]
        else:
            batch_size = int(batch.shape[1])
            lengths = (batch.sum(dim=2) != 0.0).sum(dim=0)
            embedded_inputs = batch.to(self.device)

        return batch_size, lengths, embedded_inputs

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Passes the embedded inputs through the reservoir.
        In this case, there is no reservoir, hence nothing is done (except a transpose).
        This method will be overwritten by the next children classes.

        Parameters
        ----------
        batch_size : `int`
            Batch size.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.
        embedded_inputs : `torch.Tensor`
            3D tensor [max length x batch size x embedding dim] containing the embedded inputs.

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) states is a 3D tensor [batch size x max length x embedding dim] containing the reservoir states.
            (ii) lengths is a 1D tensor [batch size].
        """
        states = embedded_inputs.transpose(0, 1)

        return states, lengths

    def forward(self, batch):
        """
        Implements the forward pass per se.
        Passes the input batch through the reservoir (using the ``_forward`` method).

        Parameters
        ----------
        batch : `transformers.tokenization_utils_base.BatchEncoding` or `torch.Tensor`
            Input batch to be processed.
            Either a Hugging Face batch of tokenized texts;
            Or a 3d tensor of states [max length x batch size x embedding dim]

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) states is a 3D tensor [batch size x max length x embedding dim] containing the reservoir states.
            (ii) lengths is a 1D tensor [batch size].
        """
        return self._forward(*self._embed(batch))

    def reverse_forward(self, batch):
        """
        Implements the forward pass, but with the texts or states in the reversed order.
        Reverses the inout batch and passes through the reservoir (using the ``forward`` method).

        Parameters
        ----------
        batch : `transformers.tokenization_utils_base.BatchEncoding` or `torch.Tensor`
            Input batch to be processed.
            Either a Hugging Face batch of tokenized texts;
            Or a 3d tensor of states [max length x batch size x embedding dim]

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) states is a 3D tensor [batch size x max length x embedding dim] containing the reservoir states.
            (ii) lengths is a 1D tensor [batch size].
        """
        # flip batch
        batch_rev = batch.copy()
        batch_rev["input_ids"] = batch["input_ids"].clone()

        for i, l in enumerate(batch["lengths"]):
            batch_rev["input_ids"][i, :l] = torch.flip(batch["input_ids"][i, :l], dims=[0])

        # pass reversed batch through the layer
        return self.forward(batch_rev)

    def warm_up(self, warm_up_sequence, return_states=False):
        """
        Warms up the ESN (in the case its reservoir is built with a ``LayerRecurrent``).
        Passes successive sequences through the ESN and updates its initial state.
        In this case, there is no reservoir, hence nothing is done.
        This method will be overwritten by the next children classes.

        Parameters
        ----------
        warm_up_sequence : transformers.tokenization_utils_base.BatchEncoding
            batch of sentences used for the warm up.
        """
        raise NotImplementedError("Warm up should probably not be used with this kind of layer.")


class LayerLinear(Layer):
    """
    Implements a **linear layer** of a network.
    A linear layer is composed of an **embedding** and a **linear reservoir** of neurons:

    LINEAR LAYER = EMBEDDING + LINEAR RESERVOIR.

    This layer takes a tokenized text as input,
    embeds it, and pass it through a linear layer - or an non-recurrent reservoir -
    (possibly in both directions).
    This class inherits from `Layer`.

    Parameters
    ----------
    dim : `int`
        Dimension of the layer.
    input_scaling : `float`
        Input scaling: bounds of the uniform distribution from which the input weights are drawn.
    bias_scaling : `float`
        Bias scaling: bounds of the uniform distribution from which the biases are drawn.
    activation_function : `str`
        Activation function of the neurons: either `tanh` or `relu`.
    **kwargs : optional
        Other keyword arguments.
    """

    def __init__(self,
                 dim=None,
                 input_scaling=None,
                 bias_scaling=None,
                 activation_function='tanh',
                 **kwargs
                 ):

        super().__init__(**kwargs)

        # scaling and dimension
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.dim = dim

        # activation function
        if activation_function == 'tanh':
            self.activation_function = torch.tanh
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            print("Activation function unknown...")
            self.activation_function = None

        # input weights
        input_w = mat.generate_uniform_matrix(size=(self.dim, self.input_dim),
                                              scaling=self.input_scaling)
        input_w = Variable(input_w, requires_grad=False)
        self.register_buffer('input_w', input_w)

        # bias
        if self.bias_scaling is not None:
            bias = mat.generate_uniform_matrix(size=(1, self.dim),
                                               scaling=self.bias_scaling).flatten()
        else:
            bias = torch.zeros(size=(1, self.dim)).flatten()
        bias = Variable(bias, requires_grad=False)
        self.register_buffer('bias', bias)

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Passes the embedded inputs through the linear reservoir.
        This method overwrites that of the class `Layer`.

        Parameters
        ----------
        batch_size : `int`
            Batch size.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.
        embedded_inputs : `torch.Tensor`
            3D tensor [max length x batch size x embedding dim] containing the embedded inputs.

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) states is a 3D tensor [batch size x max length x embedding dim] containing the reservoir states.
            (ii) lengths is a 1D tensor [batch size].
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

        # # Possible loop-free implementation (to be checked)
        # # repeat input_w and bias along the batch dimension
        # input_w = self.input_w.repeat(batch_size, 1, 1).transpose(1,2)
        # bias = self.bias.repeat(batch_size, sentence_dim, 1)
        # states = self.activation_function(torch.bmm(embedded_inputs, input_w) + bias)

        return states, lengths


class LayerRecurrent(LayerLinear):
    """
    Implements a **recurrent layer** of a network.
    A recurrent layer is composed of an **embedding** and a (recurrent) **reservoir** of neurons:

    RECURRENT LAYER = EMBEDDING + RESERVOIR.

    This layer takes a tokenized text as input,
    embeds it, and pass it through a recurrent reservoir (possibly in both directions).
    This class inherits from `Layer`.

    Parameters
    ----------
    distribution : `str`
        Distribution from which the reservoir weights are drawn: 'uniform' or 'gaussian'.
    mean : `float`
        If Gaussian, mean of the distribution; ignored otherwise.
    std : `float`
        If Gaussian, standard deviation of the distribution; ignored otherwise.
    sparsity : `float`
        Number between 0 and 1 representing the percentage of reservoir weights that are 0.
    spectral_radius : `float`
        Spectral radius of the reservoir weight matrix.
    leaking_rate :
        Leaking rate of the network: between 0 and 1.
    **kwargs : optional
        Other keyword arguments.
    """

    def __init__(self,
                 distribution='gaussian',
                 mean=0.0,
                 std=1.0,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 **kwargs):

        super().__init__(**kwargs)

        # distribution, mean, std, sparsity, spectral_radius, leaking_rate
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate

        # reservoir weights
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

        # initial state
        self.register_buffer('initial_state', torch.rand(self.dim, requires_grad=False))

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Passes the embedded inputs through the reservoir.
        See the dynamical equation of an ESN in the documentation.
        This method overwrites that of the class `Layer`.

        Parameters
        ----------
        batch_size : `int`
            Batch size.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.
        embedded_inputs : `torch.Tensor`
            3D tensor [max length x batch size x embedding dim] containing the embedded inputs.

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) states is a 3D tensor [batch size x max length x embedding dim] containing the reservoir states.
            (ii) lengths is a 1D tensor [batch size].
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

    def warm_up(self, warm_up_sequence):
        """
        Warms up the ESN.
        Passes successive sequences through the ESN and updates its initial state.

        Parameters
        ----------
        warm_up_sequence : transformers.tokenization_utils_base.BatchEncoding
            batch of sentences used for the warm up.
        """

        # pass texts through the layer
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)
        # Remove first dimension and take last valid state
        self.initial_state = warm_states[0, warm_sentence_length - 1, :].reshape(-1)


def create_layer(mode='recurrent_layer', **kwargs):
    """
    Creates a recurrent, linear or empty layer depending on the given mode.
    The possible modes are: ``'recurrent_layer'``,  ``'linear_layer'``, and ``'no_layer'``.

    Parameters
    ----------
    mode : `str`
    **kwargs : optional
        Other keyword arguments.

    Returns
    -------
    `Layer` or `ValueError`
        The layer built.
    """
    if mode == 'recurrent_layer':
        return LayerRecurrent(**kwargs)
    elif mode == 'linear_layer':
        return LayerLinear(**kwargs)
    elif mode == 'no_layer':
        return Layer(**kwargs)
    else:
        raise ValueError('wrong mode')


def get_parameters(nb_layers=1, index=0, **kwargs):
    """
    Given arguments for a deep ESN with n reservoirs, retrieves the arguments for the i-th reservoir,
    where n = ``nb_layers`` and i = ``index``.

    Parameters
    ----------
    nb_layers : `int`
    index : `int`
    **kwargs : optional
        Other keyword arguments.

    Returns
    -------
    new_kwargs : `dict`
        Dictionary of arguments for the i-th reservoir of a deep ESN with n reservoirs,
        where i = index and n = nb_layers.
    """
    new_kwargs = {}

    for key, value in kwargs.items():

        # if list of arg values (one value for each reservoir)
        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) != nb_layers:
                raise TypeError('Number of parameters and number of layers do not match...')
            # then the i-th arg value is taken (i is the current reservoir index)
            else:
                new_kwargs[key] = value[index]

        elif value is None:
            pass

        # if only one arg value, then take it for all reservoirs
        else:
            new_kwargs[key] = value

    # reservoir of have no embedding
    if index > 0:
        new_kwargs['embedding'] = None

    return new_kwargs


class DeepLayer(Layer):
    """
    Implements a deep layer, to be used in the context of a deep ESN.
    Parameters are self-explanatory.

    Implements a **deep layer** of a network.
    A deep layer is composed of an **embedding** and a succession of  **reservoir** of neurons:

    DEEP LAYER = EMBEDDING + RESERVOIR + RESERVOIR + ... + RESERVOIR.

    This layer takes a tokenized text as input,
    embeds it, and pass it through the successive reservoirs (possibly in both directions).
    This class inherits from `Layer`.

    Parameters
    ----------
    nb_layers : `int`
        Number of layers (reservoirs).
    **kwargs : optional
        Other keyword arguments.
    """

    def __init__(self, nb_layers=1, **kwargs):

        super().__init__(**kwargs)

        self.nb_layers = nb_layers

        # generate the reservoirs composing the deep layer
        self.layers = []

        for i in range(self.nb_layers):
            if i > 0:
                kwargs['input_dim'] = self.layers[i - 1].dim  # dim of R_{i} = input_dim of R_{i+1}
            # get params of reservoir i and create layer i accordingly
            layer = create_layer(**get_parameters(self.nb_layers, i, **kwargs))
            self.layers.append(layer)

        # Put buffer variables of the layers on device
        for layer in self.layers:
            for k, v in layer.__dict__['_buffers'].items():
                layer.__dict__['_buffers'][k] = v.to(self.device)

    def _forward(self, batch_size, lengths, embedded_inputs):
        """
        Passes the embedded inputs through the successive reservoirs and concatenates them.
        See the dynamical equation of a deep ESN for further details.
        This method overwrites that of the class `Layer`.

        Parameters
        ----------
        batch_size : `int`
            Batch size.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.
        embedded_inputs : `torch.Tensor`
            3D tensor [max length x batch size x embedding dim] containing the embedded inputs.

        Returns
        -------
        `tuple` [`torch.Tensor`, `torch.Tensor`]
            (states, lengths):
            (i) concatenated_states is a 3D tensor [batch size x max length x embedding dim * nb reservoirs]
            containing the deep reservoir states.
            (ii) lengths is a 1D tensor [batch size].
        """

        # Initial input: batch of texts
        current_inputs = embedded_inputs

        # Forward pass through all reservoirs
        states_l = []

        for layer in self.layers:
            states, lengths = layer._forward(batch_size, lengths, current_inputs)
            states_l.append(states)
            current_inputs = states.transpose(0, 1)

        concatenated_states = torch.cat(states_l, dim=2)

        return concatenated_states, lengths

    def warm_up(self, warm_up_sequence):
        """
        Warms up the deep ESN.
        Passes successive sequences through the deep ESN and updates the initial state of all reservoirs.

        Parameters
        ----------
        warm_up_sequence : transformers.tokenization_utils_base.BatchEncoding
            batch of sentences used for the warm up.
        """

        # Process text into the sequence of reservoirs
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)

        # set new initial states of each layer
        last_state, index = warm_states[-1, -1, :], 0

        for layer in self.layers:
            dim = layer.dim
            layer.initial_state = last_state[index: index + dim]
            index = index + dim
