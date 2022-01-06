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


class Reservoir(nn.Module):
    """
    Implements the reservoir of an echo state network (ESN).
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
    sparsity : float
        Sparsity of the reservoir ((between 0 and 1))
    spectral_radius : float
        Spectral radius of the reservoir weights.
        Should theoretically be below 1, but slightly above 1 works in practice.
    leaking_rate : float (between 0 and 1)
        Leaking rate of teh reservoir (between 0 and 1).
        Determines the amount of last state and current input involved in the current state updating.
    activation_function : str
        Activation function of the reservoir cells ('tanh' by default).
    seed : torch._C.Generator
        Random seed.
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 reservoir_dim=None,
                 bias_scaling=None,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=None,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu')):

        super(Reservoir, self).__init__()

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
        elif embedding_weights is None:
            self.embedding = None

        # Random seed
        torch.manual_seed(seed)
        self.seed = seed
        
        # Scalings
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        # Input and reservoir dim
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim

        # Sparsity, spectral_radius
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate

        # Activation function
        if activation_function == 'tanh':
            self.activation_function = torch.tanh
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            print("Activation function unknown...")
            self.activation_function = None

        # Device: CPU or GPU
        self.device = device

    def forward(self, batch):
        """
        Implements the processing of a batch of input texts into the reservoir.

        Parameters
        ----------
        batch : Union[transformers.tokenization_utils_base.BatchEncoding, torch.Tensor]
            In the case of a dynamic embedding, like BERT, 
            batch is a BatchEncoding object output by a Hugging Face tokenizer.
            Usually, batch contains different keys, like 'attention_mask', 'input_ids', 'labels', 'lengths'...
            batch['input_ids'] is a 2D tensor (batch_size x max_length) of the form:
            tensor([[ 101, 2129, 2001,  ...,    0,    0,    0],
                    [ 101, 2054, 2003,  ...,  102,    0,    0],
                    [ 101, 2073, 2003,  ...,  102,    0,    0],
                    ...,
                    [ 101, 2054, 2001,  ..., 7064, 1029,  102],
                    [ 101, 2054, 2024,  ..., 2015, 1029,  102],
                    [ 101, 2073, 2003,  ..., 2241, 1029,  102]])
            This tensor is composed by the tokenized sentences of the batch stacked horizontally.
            In the case of a static embedding, batch is a 2D tensor (batch_size x max_length) of the form:
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
            states : 3D tensor (batch_size x max_length x reservoir_dim).
            Reservoir states obtained after processing the batch of inputs into the reservoir.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        """

        # We distinguish 2 cases:
        # case 1: input batch is a 2D tensor of word indices (input ids to be embedded)
        # case 2: input batch is a 3D tensor of states (states of previous reservoir, used in deep ESNs)

        # case 1
        if callable(self.embedding):
            batch_size = int(batch["input_ids"].shape[0])
            # -2 because we will ignore the first [CLS], and the last [SEP] 
            lengths = batch["lengths"].to(self.device) - 2
            embedded_inputs = self.embedding(batch)
            #Ignore the first [CLS]
            embedded_inputs = embedded_inputs[1:, :, :]
        # case 2
        elif self.embedding is None:
            batch_size = int(batch.size()[0])
            lengths = batch.sum(dim=2).shape[1] - (batch.sum(dim=2) == 0.0).sum(dim=1)
            embedded_inputs = torch.transpose(batch, 0, 1)

        # States: left uninitialized to speed up things
        states = torch.empty(batch_size, lengths.max(), self.reservoir_dim, dtype=torch.float32, device=self.device)

        # Set initial reservoir state
        current_reservoir_states = self.initial_state.expand(batch_size, -1).transpose(0, 1)
        
        # For each time step, we process all sentences in the batch concurrently
        for t in range(lengths.max()):

            # Current input (embedded word)
            u = embedded_inputs[t, :, :]

            # Input activation
            u_act = torch.mm(self.input_w, u.transpose(0, 1))

            # Current reservoir state
            x = current_reservoir_states

            # Reservoir activation
            x_act = torch.mm(self.reservoir_w, x)

            # New reservoir state f(input_act + reservoir_act)
            x_new = self.activation_function(u_act + x_act + self.bias.expand(batch_size, -1).transpose(0, 1))

            x_new = (1 - self.leaking_rate) * x + self.leaking_rate * x_new

            # Amazing one-liner to zero any reservoir state after the respective sentence end
            x_new *= (lengths - t > 0)

            # Add new reservoir state to states
            states[:, t, :] = x_new.transpose(0, 1)

            # New reservoir state becomes current reservoir state
            current_reservoir_states = x_new
                
        return states, lengths

    def reverse_forward(self, batch):
        """
        This function return the reservoir states obtained when the tokens ids
        are passed through the reservoir in the inverse order.
        It applies the forward on a reversed batch.
        Note that it will not inverse the padding tokens.

        Parameters
        ----------
        batch : Union[transformers.tokenization_utils_base.BatchEncoding, torch.Tensor]
            3D tensor. See the docstring of the forward method for further description.

        Returns
        -------
        reversed_states : torch.Tensor
            3D tensor: batch of states obtained after the processing of the reversed input.
            The padded states are kept to their initial locations.
        lengths : torch.Tensor
            1D tensor: see the docstring of the forward method for further description.
        """

        # flip tensor (cf. Yves code)
        batch_rev = batch.copy()
        batch_rev["input_ids"] = batch["input_ids"].clone()
        
        for i, l in enumerate(batch["lengths"]):
            batch_rev["input_ids"][i, :l] = torch.flip(batch["input_ids"][i, :l], dims=[0])
        
        # make the reversed batch go through the reservoir
        states_rev, lengths = self.forward(batch_rev)
        
        return states_rev, lengths
    

    def warm_up(self, warm_up_sequence, mode=None):
        """
        Performs forward pass of an input sequence and set last reservoir state as new initial state.

        Parameters
        ----------
        warm_up_sequence : torch.Tensor
            1D tensor: word indices of the warm up sentence.
        """

        if not callable(self.embedding):  # TorchText
            # Add a first dimension to sequence to match 2D input batch format
            warm_up_sequence = warm_up_sequence.unsqueeze(1)

        # Process text into the reservoir
        warm_states, warm_sentence_length = self.forward(warm_up_sequence)
        # Remove first dimension and take last valid state
        self.initial_state = warm_states[0, warm_sentence_length - 1, :].reshape(-1)
        
        if mode == 'return_states':
            self.warm_states = warm_states


class UniformReservoir(Reservoir):
    """
    Implements a reservoir generated from a uniform distribution.

    Parameters
    ----------
    embedding_weights : torch.Tensor
    input_dim : int
    input_scaling : float
    reservoir_dim : int
    bias_scaling : float
    sparsity : float
    spectral_radius : float
    leaking_rate : float
    activation_function : str
    seed : torch._C.Generator
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 reservoir_dim=None,
                 bias_scaling=None,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu')):

        super(UniformReservoir, self).__init__(embedding_weights,
                                               input_dim,
                                               input_scaling,
                                               reservoir_dim,
                                               bias_scaling,
                                               sparsity,
                                               spectral_radius,
                                               leaking_rate,
                                               activation_function,
                                               seed,
                                               device)
        
        # Distribution
        self.distribution = 'uniform'
        
        # Inputs
        input_w = mat.generate_uniform_matrix(size=(self.reservoir_dim, self.input_dim),
                                              scaling=self.input_scaling)
        input_w = Variable(input_w, requires_grad=False)
        self.register_buffer('input_w', input_w)
        
        # Reservoir
        reservoir_w = mat.generate_uniform_matrix(size=(self.reservoir_dim, self.reservoir_dim),
                                                  scaling=self.input_scaling,
                                                  sparsity=self.sparsity,
                                                  spectral_radius=self.spectral_radius)
        reservoir_w = Variable(reservoir_w, requires_grad=False)
        self.register_buffer('reservoir_w', reservoir_w)

        # Bias
        if self.bias_scaling is not None:
            bias = mat.generate_uniform_matrix(size=(1, self.reservoir_dim), 
                                               scaling=self.bias_scaling).flatten()
        else:
            bias = torch.zeros(size=(1, self.reservoir_dim)).flatten()
        bias = Variable(bias, requires_grad=False)
        self.register_buffer('bias', bias)

        # Initial_state
        self.register_buffer('initial_state', torch.rand(self.reservoir_dim, requires_grad=False))


class GaussianReservoir(Reservoir):
    """
    Implements a reservoir generated from a Gaussian distribution.

    Parameters
    ----------
    embedding_weights : torch.Tensor
    input_dim : int
    input_scaling : float
    reservoir_dim : int
    bias_scaling : float
    sparsity : float
    spectral_radius : float
    leaking_rate : float
    mean : float
    std : float
    activation_function : str
    seed : torch._C.Generator
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 reservoir_dim=None,
                 bias_scaling=None,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 mean=0.0,
                 std=1.0,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu')):

        super(GaussianReservoir, self).__init__(embedding_weights,
                                                input_dim,
                                                input_scaling,
                                                reservoir_dim,
                                                bias_scaling,
                                                sparsity,
                                                spectral_radius,
                                                leaking_rate,
                                                activation_function,
                                                seed,
                                                device)

        # Distribution, mean, std
        self.distribution = 'gaussian'
        self.mean = mean
        self.std = std
        
        # Inputs (these stay uniform)
#         input_w = mat.generate_gaussian_matrix(size=(self.reservoir_dim, self.input_dim),
#                                                mean=self.mean,
#                                                std=self.std)
#         input_w = Variable(input_w, requires_grad=False)
#         self.register_buffer('input_w', input_w) 
        input_w = mat.generate_uniform_matrix(size=(self.reservoir_dim, self.input_dim),
                                              scaling=self.input_scaling)
        input_w = Variable(input_w, requires_grad=False)
        self.register_buffer('input_w', input_w)
        
        
        # Reservoir
        reservoir_w = mat.generate_gaussian_matrix(size=(self.reservoir_dim, self.reservoir_dim),
                                                   sparsity=self.sparsity,
                                                   mean=self.mean,
                                                   std=self.std,
                                                   spectral_radius=self.spectral_radius)
        reservoir_w = Variable(reservoir_w, requires_grad=False)
        self.register_buffer('reservoir_w', reservoir_w)

        # Bias
        if self.bias_scaling is not None:
            bias = mat.generate_uniform_matrix(size=(1, self.reservoir_dim), 
                                               scaling=self.bias_scaling).flatten()
        else:
            bias = torch.zeros(size=(1, self.reservoir_dim)).flatten()
        bias = Variable(bias, requires_grad=False)
        self.register_buffer('bias', bias)

        # Initial_state
        self.register_buffer('initial_state', torch.rand(self.reservoir_dim, requires_grad=False))


class NoneReservoir(Reservoir):
    """
    Implements the absence of reservoir.
    Used for implementing baselines, which precisely consist of ESN whose reservoir is omitted.

    Parameters
    ----------
    embedding_weights : torch.Tensor
    seed : torch._C.Generator
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 reservoir_dim=None,
                 bias_scaling=None,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu')):

        # specify all attributes from Reservoir for proper initialization
        super(NoneReservoir, self).__init__(embedding_weights=embedding_weights,
                                            input_dim=None,
                                            reservoir_dim=None,
                                            bias_scaling=None,
                                            sparsity=None,
                                            spectral_radius=None,
                                            leaking_rate=None,
                                            activation_function=None,
                                            seed=seed,
                                            device=device)

    # Overriding forward method for the no reservoir case
    def forward(self, batch):
        """
        Simply gather batches of embedded inputs: no reservoir processing.

        Parameters
        ----------
        inputs : torch.Tensor
            2D input tensor (max_length x batch_size).
            A batch of input texts is a 2D tensor.
            Each tensor column represents a text - given as the sequence of its word indices.

        Returns
        -------
        embedded_inputs, lengths : torch.Tensor, torch.Tensor
            embedded_inputs : 3D tensor (batch_size x max_length x reservoir_dim).
            Embedded inputs are simply obtained by embedding batch of inputs.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
        """

        lengths = batch["lengths"].to(self.device)
        embedded_inputs = self.embedding(batch).transpose(0, 1)

        return embedded_inputs, lengths

    
class ReservoirFC(UniformReservoir):
    """
    Implements a resevoir that consists of a fully connected layer.
    Used for implementing the Custom Baseline, which consists of:
    - an embedding layer (EMB)
    - followed by a fully connected layer (FC) 
    - followed by a learning algorithm (LA) (e.g., Ridge regression).
    Note that the FC layer is not trained.
    Custom RR-Baseline = EMB + FC + LA

    Parameters
    ----------
    embedding_weights : Union[torch.Tensor, str]
        The embedding weights can be of 2 kinds: static or dynamic:
        A static embedding is usually specified by a corresponding lookup matrix (torch.Tensor).
        The dynamic embeddings are those generated by transformer-based models, like BERT for instance.
        A dynamic embedding is usually specified by its name, e.g, "bert-base-uncased" (str).
    input_dim : int
        Input dimension.
    reservoir_dim : int
        Reservoir dimension.
    input_scaling : float
        Input scaling: bounds used for the uniform random generation of the input weights.
    bias_scaling : float
        Bias scaling: bounds used for the uniform random generation of the bias weights.
    activation_function : str
        Activation function of the reservoir cells ('tanh' by default).
    seed : torch._C.Generator
    """

    # Constructor
    def __init__(self,
                 embedding_weights=None,
                 input_dim=None,
                 input_scaling=None,
                 reservoir_dim=None,
                 bias_scaling=None,
                 sparsity=None,
                 spectral_radius=None,
                 leaking_rate=1.0,
                 activation_function='tanh',
                 seed=42,
                 device=torch.device('cpu')):

        # specify all attributes from Reservoir for proper initialization
        super(ReservoirFC, self).__init__(embedding_weights=embedding_weights,
                                          input_dim=input_dim,
                                          input_scaling=input_scaling,
                                          reservoir_dim=reservoir_dim,
                                          bias_scaling=bias_scaling,
                                          sparsity=sparsity,
                                          spectral_radius=spectral_radius,
                                          leaking_rate=leaking_rate,
                                          activation_function=activation_function,
                                          seed=seed,
                                          device=device)
        
    # Overriding forward method for this unrecurrent reservoir
    def forward(self, batch):
        """
        Simply gather batches of embedded inputs: no reservoir.

        Parameters
        ----------
        batch : Union[transformers.tokenization_utils_base.BatchEncoding, torch.Tensor]
            In the case of a dynamic embedding, like BERT, 
            batch is a BatchEncoding object output by a Hugging Face tokenizer.
            Usually, batch contains different keys, like 'attention_mask', 'input_ids', 'labels', 'lengths'...
            batch['input_ids'] is a 2D tensor (batch_size x max_length) of the form:
            tensor([[ 101, 2129, 2001,  ...,    0,    0,    0],
                    [ 101, 2054, 2003,  ...,  102,    0,    0],
                    [ 101, 2073, 2003,  ...,  102,    0,    0],
                    ...,
                    [ 101, 2054, 2001,  ..., 7064, 1029,  102],
                    [ 101, 2054, 2024,  ..., 2015, 1029,  102],
                    [ 101, 2073, 2003,  ..., 2241, 1029,  102]])
            This tensor is composed by the tokenized sentences of the batch stacked horizontally.
            In the case of a static embedding, batch is a 2D tensor (batch_size x max_length) of the form:
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
        embedded_inputs, lengths : torch.Tensor, torch.Tensor
            embedded_inputs : 3D tensor (batch_size x max_length x reservoir_dim).
            Embedded inputs is the 3D tensor obtained after embedding the batch of input tokens.
            lengths : 1D tensor (batch_size).
            Lengths of input texts in the batch.
            Note that the embeddings of the tokens [CLS] and [SEP] generated by 
            the Hugging Face tokenizer are discarded from the embedded_inputs tensor.
        """
        
        # FC layer
        self.fc = nn.Linear(self.input_dim, self.reservoir_dim).to(self.device)
        
        batch_size = int(batch["input_ids"].shape[0])
        lengths = batch["lengths"].to(self.device)
        
        embedded_inputs = self.embedding(batch).transpose(0, 1)
                
        if callable(self.embedding):
            lengths = lengths - 2
            # Ignore the first [CLS] token
            embedded_inputs = embedded_inputs[:, 1:, :]
            sentence_dim = int(embedded_inputs.shape[1])
        
        # repeat input_w and bias along the batch dimension
        input_w = self.input_w.repeat(batch_size, 1, 1).transpose(1,2)
        bias = self.bias.repeat(batch_size, sentence_dim, 1)
        # x = f(W_in u + b)
        states = self.activation_function(torch.bmm(embedded_inputs, input_w) + bias)
                        
        return states, lengths
