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
from transformers import BatchEncoding
import esntorch.utils.matrix as mat
import esntorch.core.reservoir as res
import esntorch.core.learning_algo as la
import esntorch.core.pooling_strategy as ps
from tqdm.notebook import tqdm_notebook


class EchoStateNetwork(nn.Module):
    """
    Implements an **echo state Nnetwork (ESN)** per se.
    An ESN consists of the combination of
    a **layer**, a **pooling strategy** and a **learning algorithm**:

    ESN = LAYER + POOLING + LEARNING ALGO.

    Recalling that a general **layer** is itslef composed of
    an **embedding** and a **reservoir** of neurons, one finally has:

    ESN = EMBEDDING + RESERVOIR + POOLING + LEARNING ALGO.

    Parameters
    ----------
    learning_algo : `object`
        A learning algo used to train the netwok (see learning_algo.py).
    criterion : `torch.nn.Module` or `None`
        Pytorch loss used to train the ESN (in case of non-direct methods).
    optimizer : `torch.optim.Optimizer` or `None`
        Pytorch optimizer used to train the ESN (in case of non-direct methods).
    pooling_strategy : `str`
        Pooling strategy to be used: 'mean', 'first', 'last',
        'weighted', 'lexicon_weighted', None,
    bidirectional : `bool`
        Whether to implement a bi-directional ESN or not,
    lexicon : `torch.Tensor` or `None`
        If not None, lexicon to be used with the pooling strategy 'lexicon_weighted'.
    deep : `bool`
        Whether to implement a deep ESN or not.
    device : `int`
        The device to be used: cpu or gpu.
    **kwargs : optional
        Other keyword arguments.
    """

    def __init__(self,
                 learning_algo=None,
                 criterion=None,
                 optimizer=None,
                 pooling_strategy=None,
                 bidirectional=False,
                 lexicon=None,
                 deep=False,
                 device=torch.device('cpu'),
                 **kwargs
                 ):

        super().__init__()
        self.device = device

        if deep:
            self.layer = res.DeepLayer(device=device, **kwargs)
        else:
            self.layer = res.create_layer(device=device, **kwargs)

        self.pooling_strategy = ps.Pooling(pooling_strategy, lexicon=lexicon)

        self.learning_algo = learning_algo
        self.criterion = criterion
        self.optimizer = optimizer
        self.bidirectional = bidirectional

    def warm_up(self, dataset):
        """
        Warms up the ESN (in the case its reservoir is built with a ``LayerRecurrent``).
        Passes successive sequences through the ESN and updates its initial state.
        In this case, there is no reservoir, hence nothing is done.
        This method will be overwritten by the next children classes.

        Parameters
        ----------
        dataset : `datasets.arrow_dataset.Dataset`
            Datasets of sentences to be used for the warming.
        """

        for i, text in enumerate(dataset):

            for k, v in text.items():
                text[k] = v.unsqueeze(0)

            text = BatchEncoding(text)
            self.layer.warm_up(text)

    def _apply_pooling_strategy(self, states, lengths, texts,
                                reversed_states=None, additional_fts=None):
        """
        Pools the reservoir states according to a pooling strategy.
        If the ESN is bi-directional, the reversed states are also used.

        Parameters
        ----------
        states : `torch.Tensor`
            3D tensor [batch size x max length x reservoir dim]
            containing the reservoir states.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.
        texts: `transformers.tokenization_utils_base.BatchEncoding`
            Batch of token ids.
        reversed_states : `torch.Tensor` or `None`
            3D tensor [batch size x max length x reservoir dim]
            containing the reversed reservoir states.
        additional_fts : `torch.Tensor` or `None`
            In the case of 'mean_and_additional_fts' pooling.
            2D tensor [batch size x fts dim] containing the additional features
            to be concatenated to the merged states.

        Returns
        -------
        final_states : torch.Tensor
            2D tensor [batch size x reservoir dim] containing the merged states.
        """
        if self.bidirectional:
            if self.pooling_strategy.pooling_strategy is None:
                # first concatenate the normal and reversed states along the layer dimension, then apply the pooling.
                restored_states = reversed_states.clone()
                for i, l in enumerate(lengths):
                    restored_states[i, :l] = torch.flip(restored_states[i, :l], [0])

                concatenated_states = torch.cat([states, restored_states], dim=2)
                final_states = self.pooling_strategy(concatenated_states, lengths, texts)
            else:
                # first pool the states and reverse states, then concatenate the merged states
                normal_merged_states = self.pooling_strategy(states, lengths, texts, additional_fts)
                reversed_merged_states = self.pooling_strategy(reversed_states, lengths, texts)
                # concatenate batches across features dimension
                final_states = torch.cat([normal_merged_states, reversed_merged_states], dim=1)
        else:
            final_states = self.pooling_strategy(states, lengths, texts, additional_fts)

        return final_states

    def _fit_direct(self, train_dataloader):
        """
        Fits ESN with direct method (i.e., no gradient descent).
        This fit is to be used with the following learning algorithms:
        ``RidgeRegression``, ``RidgeRegression_skl``, ``LinearSVC``, and
        ``LogisticRegression_skl``.

        Parameters
        ----------
        train_dataloader: `torch.utils.data.dataloader.DataLoader`
            Training dataloader.
        """

        print("Computing closed-form solution...")

        states_l = []
        labels_l = []

        # loop over batches
        for i, batch in enumerate(tqdm_notebook(train_dataloader)):

            batch_text = batch
            batch_label = batch["labels"].to(self.device)
            if 'additional_fts' in batch.keys():
                additional_fts = batch["additional_fts"].to(self.device)
            else:
                additional_fts = None

            # forward pass
            states, lengths = self.layer.forward(batch_text)  # states

            # forward pass with reversed states
            reversed_states = None
            if self.bidirectional:
                reversed_states, _ = self.layer.reverse_forward(batch_text)

            labels = batch_label

            # if pooling_strategy is None, then duplicate labels
            if self.pooling_strategy.pooling_strategy is None:
                labels = mat.duplicate_labels(labels, lengths)

            # apply pooling
            final_states = self._apply_pooling_strategy(states, lengths, batch_text,
                                                        reversed_states, additional_fts)
            states_l.append(final_states)
            labels_l.append(labels)

        all_states, all_labels = torch.cat(states_l, dim=0), torch.cat(labels_l, dim=0)

        self.learning_algo.fit(all_states, all_labels)

        print("\nTraining complete.")

    def _fit_GD(self, train_dataloader, epochs=1, iter_steps=100):
        """
        Fits ESN with gradient descent method.
        This fit is to be used with the following learning algorithms:
        ``LogisticRegression`` and ``DeepNN``.

        Parameters
        ----------
        train_dataloader: `torch.utils.data.dataloader.DataLoader`
            Training dataloader.
        epochs : `int`
            Number of training epochs.
        iter_steps : `int`
            Number of steps (batches) after which the loss is printed.

        Returns
        -------
        loss_l: `list`
            List of training losses.
        """

        print("Performing gradient descent...")

        loss_l = []
        n_iter = 0

        # loop over epochs
        for epoch in tqdm_notebook(range(int(epochs))):

            # loop over batches
            for i_batch, batch in enumerate(tqdm_notebook(train_dataloader, leave=False)):

                batch_text = batch
                batch_label = batch["labels"].to(self.device)
                if 'additional_fts' in batch.keys():
                    additional_fts = batch["additional_fts"].to(self.device)
                else:
                    additional_fts = None

                # forward pass
                states, lengths = self.layer.forward(batch_text)  # states

                # forward pass with reversed states
                reversed_states = None
                if self.bidirectional:
                    reversed_states, _ = self.layer.reverse_forward(batch_text)

                labels = batch_label.type(torch.int64)  # labels converted to int in this case
                # if pooling_strategy is None, then duplicate labels
                if self.pooling_strategy.pooling_strategy is None:
                    labels = mat.duplicate_labels(labels, lengths)

                # apply pooling
                final_states = self._apply_pooling_strategy(states, lengths, batch_text,
                                                            reversed_states, additional_fts)

                outputs = self.learning_algo(final_states)  # outputs

                if isinstance(self.criterion, torch.nn.MultiLabelSoftMarginLoss) or \
                        isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    labels = torch.nn.functional.one_hot(labels).double()

                loss = self.criterion(outputs, labels)  # compute loss
                self.optimizer.zero_grad()  # reset optimizer gradient
                loss.backward()  # backward pass
                self.optimizer.step()  # update weights

                n_iter += 1

                if n_iter % iter_steps == 0:
                    print("Iteration: {iteration} Loss: {loss}".format(iteration=n_iter, loss=loss.item()))
                    loss_l.append(loss.item())

        print("\nTraining complete.")

        return loss_l

    def fit(self, train_dataloader, epochs=1, iter_steps=100):
        """
        Fits ESN.
        Calls the correct ``_fit_direct()`` or ``_fit_GD()`` method
        depending on the learning algorithm.
        The parameters ``epochs`` and ``iter_steps`` are ignored
        in the case of a ``_fit_direct``.

        Parameters
        ----------
        train_dataloader: `torch.utils.data.dataloader.DataLoader`
            Training dataloader.
        epochs : `int`
            Number of training epochs.
        iter_steps : `int`
            Number of steps (batches) after which the loss is printed.
        """

        # closed-form training
        if isinstance(self.learning_algo, la.RidgeRegression) or \
                isinstance(self.learning_algo, la.RidgeRegression_skl) or \
                isinstance(self.learning_algo, la.LogisticRegression_skl) or \
                isinstance(self.learning_algo, la.LinearSVC):

            return self._fit_direct(train_dataloader)

        # Gradient descent training
        elif isinstance(self.learning_algo, la.LogisticRegression) or \
                isinstance(self.learning_algo, la.DeepNN):

            return self._fit_GD(train_dataloader, epochs, iter_steps)

    def _compute_predictions(self, states, lengths):
        """
        Takes states, passes them to the learning algorithm and computes predictions out of them.
        Predictions are computed differently depending on whether the pooling strategy is None or not.
        If pooling strategy is None, the predictions are computed as follows:

        For each input sentence u:
            - the states X_u are passed through the learning algorithm;
            - the raw outputs of the algorithm Y_u are then averaged row-wise, yielding a 1-dim tensor y_u;
            - prediction = arg_max(y_u).

        If pooling strategy is not None, the predictions are computed as follows:

        For each input sentence u:
            - the merged layer state x_u is passed through the learning algorithm;
        yielding a 1-dim tensor y_u;
            - prediction = arg_max(y_u).

        Parameters
        ----------
        states : `torch.Tensor`
            3D tensor [batch size x max length x reservoir dim]
            containing the reservoir states.
        lengths : `torch.Tensor`
           1D tensor [batch size] containing the lengths of all sentences in the batch.

        Returns
        -------
        predictions : `torch.Tensor`
            Predictions computed from the outputs.
        """

        raw_outputs = self.learning_algo(states)

        # pooling strategy is None
        if self.pooling_strategy.pooling_strategy is None:

            tmp = list(lengths.cpu().numpy())
            tmp = [0] + [sum(tmp[:i]) for i in range(1, len(tmp) + 1)]
            outputs = torch.stack([torch.mean(raw_outputs[tmp[i]:tmp[i + 1]], dim=0) for i in range(len(tmp) - 1)])
            predictions = outputs.argmax(dim=1)

        # pooling strategy is not None
        else:
            if raw_outputs.dim() != 1:  # the learning algo returns the probs
                outputs = raw_outputs.argmax(dim=1).float()
            else:  # the learning algo returns the classes
                outputs = raw_outputs.float()
            predictions = outputs.type(torch.int64)

        return predictions

    def predict(self, dataloader, verbose=True):
        """
        Evaluates the ESN on a dataloader (train, test, validation).
        Returns the list of prediction labels.
        If the true labels are known, then returns the accuracy also.

        Parameters
        ----------
        dataloader : `torch.utils.data.dataloader.DataLoader`
            Dataloader.
        verbose : `bool`

        Returns
        -------
        `tuple` [`list`, `float`]
            (predictions, accuracy): list of predictions and accuracy.
        """

        predictions_l = []
        correct = 0
        total = 0
        testing_mode = False

        for i, batch in enumerate(tqdm_notebook(dataloader)):

            if callable(self.layer.embedding):  # HuggingFace # XXX fix because no more else
                batch_text = batch
                batch_label = batch["labels"].to(self.device)
                if 'additional_fts' in batch.keys():
                    additional_fts = batch["additional_fts"].to(self.device)
                else:
                    additional_fts = None

            # forward pass
            states, lengths = self.layer.forward(batch_text)

            # forward pass with reversed text
            reversed_states = None
            if self.bidirectional:
                reversed_states, _ = self.layer.reverse_forward(batch_text)

            # apply pooling
            final_states = self._apply_pooling_strategy(states, lengths, batch_text,
                                                        reversed_states, additional_fts)

            predictions = self._compute_predictions(final_states, lengths)
            predictions_l.append(predictions.reshape(-1))

            # if labels available, then compute accuracy
            try:
                labels = batch_label.type(torch.int64)
                total += labels.size(0)
                correct += (predictions == labels).sum()
                testing_mode = True
            # otherwise, pure prediction mode
            except Exception:
                pass

        accuracy = 100 * correct.item() / float(total) if testing_mode else None
        predictions_l = torch.cat(predictions_l, dim=0).cpu().detach().numpy()

        if verbose and testing_mode:
            print("\nAccuracy: {}.".format(accuracy))

        return predictions_l, accuracy
