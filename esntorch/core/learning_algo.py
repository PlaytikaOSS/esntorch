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
from sklearn.linear_model import LogisticRegression as LogisticRegression_
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC as LinearSVC_


class RidgeRegression(torch.nn.Module):
    """
    Implements Ridge regression via its closed form solution:
    $$ beta = (X^T X + lambda I)^{-1} X^T y $$
    Works for multi-class problems with target values 0, 1, 2, etc.

    Parameters
    ----------
    alpha : `float`
    mode : `None` or `str`
        Default mode is None.
        The modes 'normalize' and 'standardize' are also possible.
    """

    def __init__(self, alpha=1., mode=None):

        super(RidgeRegression, self).__init__()

        self.alpha = alpha
        self.mode = mode
        self.weights = None
        self.mean = None
        self.std = None
        self.L2norm = None

    def fit(self, X, y):
        """
        Computes the closed form solution of the Ridge regression
        and update the learned weights.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).
        y : `torch.Tensor`
            Tensor of targets (gathered by rows).
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')

        # Add a column of ones to X to represent the bias
        X_ = torch.cat([X, torch.ones(X.size()[0], device=device).view(-1, 1)], dim=1)

        if self.mode == 'normalize':
            self.mean = X_.mean(dim=0)
            self.L2norm = X_.norm(p='fro', dim=0)
            X_ = torch.div(X_ - self.mean, self.L2norm)
        elif self.mode == 'standardize':
            self.mean = X_.mean(dim=0)
            self.std = X_.std(dim=0)
            X_ = torch.div(X_ - self.mean, self.std)

        y_ = torch.zeros(y.size()[0], torch.unique(y).size()[0], dtype=torch.float32, device=device)
        y_ = y_.scatter(1, y.long().unsqueeze(1), 1.0)

        self.X_ = X_
        self.y_ = y_
        LI = torch.eye(X_.size()[1], device=device) * self.alpha
        Xt = torch.transpose(X_, 0, 1)
        beta = torch.mm(Xt, X_) + LI
        beta = torch.pinverse(beta)  # compute pseudo-inverse via SVD.
        beta = torch.mm(beta, Xt)
        beta = torch.mm(beta, y_)

        self.weights = beta

    def forward(self, X):
        """
        Computes predictions.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).

        Returns
        -------
        outputs : `torch.Tensor`
            Outputs of Ridge regression
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')

        X_ = torch.cat([X, torch.ones(X.size()[0], device=device).view(-1, 1)], dim=1)

        # normalize or standardize features if needed
        if self.mode == 'normalize':
            X_ = torch.div(X_ - self.mean, self.L2norm)
        elif self.mode == 'standardize':
            X_ = torch.div(X_ - self.mean, self.std)

        outputs = torch.mm(X_, self.weights)

        return outputs


class RidgeRegression_skl:
    """
    Implements the Ridge Regression from scikit learn.
    Takes the parameters as those of sklearn.linear_model.Ridge
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
    """

    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize='deprecated',
                 copy_X=True,
                 max_iter=None,
                 tol=0.001,
                 solver='auto',
                 positive=False,
                 random_state=None):

        self.RR = RidgeClassifier(alpha=alpha,
                                  fit_intercept=fit_intercept,
                                  normalize=normalize,
                                  copy_X=copy_X,
                                  max_iter=max_iter,
                                  tol=tol,
                                  solver=solver,
                                  positive=positive,
                                  random_state=random_state)

    def fit(self, X, y):
        """
        Implements the fit method of RidgeClassifier.
        Simply converts torch tensors into numpy arrays, and applies original fit method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).
        y : `torch.Tensor`
            Tensor of targets (gathered by rows).
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
            y = y.cpu()
        X = X.numpy()
        y = y.numpy()
        self.RR.fit(X, y)

    def __call__(self, X):
        """
        Overrides parentheses method by the predict method of RidgeClassifier.
        Simply converts torch tensors into numpy arrays, and applies predict_proba method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).

        Returns
        -------
        outputs : `torch.Tensor`
            Outputs of the Ridge regression.
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
        X = X.numpy()
        outputs = self.RR.predict(X)
        outputs = torch.from_numpy(outputs).to(device)

        return outputs


class LinearSVC:
    """
    Implements Linear Support Vector Machine Classifier LinearSVC from scikit learn.
    Takes the same parameters as those of sklearn.svm.LinearSVC:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """

    def __init__(self,
                 penalty='l2',
                 loss='squared_hinge',
                 *, dual=True,
                 tol=0.0001,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=1000):

        self.SVC = LinearSVC_(penalty=penalty,
                              loss=loss,
                              dual=dual,
                              tol=tol,
                              C=C,
                              multi_class=multi_class,
                              fit_intercept=fit_intercept,
                              intercept_scaling=intercept_scaling,
                              class_weight=class_weight,
                              verbose=verbose,
                              random_state=random_state,
                              max_iter=max_iter)

    def fit(self, X, y):
        """
        Overrides the fit method of LinearSVC.
        Simply converts torch tensors into numpy arrays, and applies original fit method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).
        y : `torch.Tensor`
            Tensor of targets (gathered by rows).
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
            y = y.cpu()
        X = X.numpy()
        y = y.numpy()
        self.SVC.fit(X, y)

    def __call__(self, X):
        """
        Overrides parentheses method by predict method of LinearSVC.
        Simply converts storch tensors into numpy arrays, and applies original predict_proba method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).

        Returns
        -------
        outputs : `torch.Tensor`
            Outputs of the linear SVC regression.
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
        X = X.numpy()
        outputs = self.SVC.predict(X)
        outputs = torch.from_numpy(outputs).to(device)

        return outputs


class LogisticRegression(torch.nn.Module):
    """
    Implements classical logistic regression as a 1-layer neural network.

    Parameters
    ----------
    input_dim : `int`
       Input dimension.
    output_dim : `int`
       Output dimension.
    """

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Implements the forward pass.

        Parameters
        ----------
        x : `torch.Tensor`
            Tensor of features (gathered by rows).

        Returns
        -------
        outputs : `torch.Tensor`
            Tensor of outputs (gathered by row).
        """

        outputs = self.linear(x)

        return outputs


class LogisticRegression_skl:
    """
    Implements the Logistic Regression LogisticRegression from scikit learn.
    Takes the same parameters as those of sklearn.linear_model.LogisticRegression
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    def __init__(self,
                 penalty='l2',
                 dual=False,
                 tol=0.0001,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='liblinear',
                 max_iter=10000,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None):

        self.LR = LogisticRegression_(penalty=penalty,
                                      dual=dual,
                                      tol=tol,
                                      C=C,
                                      fit_intercept=fit_intercept,
                                      intercept_scaling=intercept_scaling,
                                      class_weight=class_weight,
                                      random_state=random_state,
                                      solver=solver,
                                      max_iter=max_iter,
                                      multi_class=multi_class,
                                      verbose=verbose,
                                      warm_start=warm_start,
                                      n_jobs=n_jobs,
                                      l1_ratio=l1_ratio)

    def fit(self, X, y):
        """
        Overrides the fit method of LogisticRegression.
        Simply converts torch tensors into numpy arrays. and applies original fit method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).
        y : `torch.Tensor`
            Tensor of targets (gathered by rows).
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
            y = y.cpu()
        X = X.numpy()
        y = y.numpy()
        self.LR.fit(X, y)

    def __call__(self, X):
        """
        Overrides parentheses method by predict method of LogisticRegression.
        Simply converts torch tensors into numpy arrays, and applies original predict_proba method.

        Parameters
        ----------
        X : `torch.Tensor`
            Tensor of features (gathered by rows).

        Returns
        -------
        outputs : `torch.Tensor`
            Outputs of Ridge regression
        """

        device = torch.device('cuda' if X.is_cuda else 'cpu')
        if device.type == 'cuda':
            X = X.cpu()
        X = X.numpy()
        outputs = self.LR.predict_proba(X)
        outputs = torch.from_numpy(outputs).to(device)

        return outputs


class DeepNN(torch.nn.Module):
    """
    Implements a deep neural network whose layers are specified in a list.
    Make sure that the dimensions of the first and last layers
    correspond to the input and output dimensions, respectively.
    Example: A 2-hidden layer NN with 50 neurons in each layer and input / output dimensions of 300 / 3
    is implemented as follows:
    layers_l = [300, 50, 50, 3];
    model = DeepNN(layers_l).

    Parameters
    ----------
    layers_l : `list` [`int`]
       List of integers representing the number on neurons in each layer.
    """

    def __init__(self, layers_l):

        super(DeepNN, self).__init__()

        self.hidden = torch.nn.ModuleList()

        for input_size, output_size in zip(layers_l, layers_l[1:]):
            self.hidden.append(torch.nn.Linear(input_size, output_size))

    def forward(self, activation):
        """
        Implements the forward pass with ReLU activation function for the hidden layers.

        Parameters
        ----------
        activation : `torch.Tensor`
            Activation values of the input layer neurons: (batch size x input dim).

        Returns
        -------
        activation : `torch.Tensor`
            Activation values of the output layer neurons: (batch size x output dim).
        """

        nb_layers = len(self.hidden)

        for (l, linear_transform) in zip(range(nb_layers), self.hidden):

            if l < nb_layers - 1:
                activation = torch.nn.functional.relu(linear_transform(activation))
                activation = torch.nn.Dropout(0.5)(activation)
            else:
                # activation = torch.nn.functional.tanh(linear_transform(activation))
                activation = linear_transform(activation)

        return activation
