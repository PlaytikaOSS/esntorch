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

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_datasets(train_dataset, test_dataset):

    # tfidf
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        analyzer='word',
        strip_accents='unicode',
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None
    )

    # train set
    train_texts = [s.text for s in train_dataset]
    train_labels = [s.label for s in train_dataset]

    # test set
    test_texts = [s.text for s in test_dataset]
    test_labels = [s.label for s in test_dataset]

    # label encoder
    le = LabelEncoder()
    le.fit(train_labels)

    # train / test labels
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    #  transformed features are sparse numpy arrays: nb_sentences x feature dimension
    train_texts_transformed = tfidf.fit_transform(train_texts)
    test_texts_transformed = tfidf.transform(test_texts)

    return {'train': (train_texts_transformed, train_labels), 'test': (test_texts_transformed, test_labels)}


def pr(x, y_i, y):
    """
    Computes feature hat{p} / norm_1(hat{p}) based on whether label y == y_i
    """
    p = x[y == y_i].sum(0)

    return (p + 1) / ((y == y_i).sum() + 1)


def pr_not(x, y_i, y):
    """
    Computes feature hat{p} / norm_1(hat{p}) based on whether label y != y_i.
    This feature is used in the 1-vs-rest version of the algorithm.
    """

    p = x[y != y_i].sum(0)

    return (p + 1) / ((y != y_i).sum() + 1)


def nbsvm(x, y, class_k=1, C=1):
    """
    Fit a NB-SVM model for a binary classification problem: class_k vs other classes
    """

    # NEW
    r = np.log(pr(x, class_k, y) / pr_not(x, class_k, y))
    m = LogisticRegression(C=C, max_iter=10000, verbose=True, solver='liblinear')
    x_nb = x.multiply(r)
    y = (y == class_k).astype(int)  # make y binary: class_k vs others

    return m.fit(x_nb, y), r


def fit(train_dataset, C=1):
    """
    In case of binary classification task:
    - fit 1 model based on features of class 1 vs features of class 2.
    In case of N-ary classification task with N>2:
    - fit N models based on features of class K vs features of classes 1,2,...,K-1,K+1,...,N, for each K = 1,...,N.
    """

    # train classes
    classes = list(set(train_dataset[1]))

    # train N models
    models_d = {}

    for class_k in classes:
        models_d[class_k] = nbsvm(train_dataset[0], train_dataset[1], class_k, C=C)     # clf, r

    return models_d


def predict(models, dataset):
    """
    Computes predictions based on all one-vs-rest trained models.
    """

    # dataset
    transformed_texts, labels = dataset[0], dataset[1]

    # predictions_scores
    prediction_scores_l = []

    for model in models.values():
        prediction_scores = model[0].predict_proba(transformed_texts.multiply(model[1]))
        prediction_scores = np.expand_dims(prediction_scores[:, 1], axis=1)   # keep only second columns: proba(X=label)
        prediction_scores_l.append(prediction_scores)

    prediction_scores = np.concatenate(prediction_scores_l, axis=1)

    # predictions
    predictions = np.argmax(prediction_scores, axis=1)

    # accuracy
    acc = accuracy_score(labels, predictions)*100

    return acc
