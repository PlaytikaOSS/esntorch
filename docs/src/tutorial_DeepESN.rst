Tutorial 2: Deep Echo State Networks for Text Classification
============================================================

This notebook presents a use case example of the ``EsnTorch`` library.
It describes the implementation of a **Deep Echo State Network (Deep
ESN)** for text classification on the **TREC-6** dataset (question
classification).

The instantiation, training and evaluation of a Deep ESN for text
classification is similar to that of a calssical ESN. It is achieved via
the following steps:

#. Import libraries and modules
#. Load and prepare data
#. Instantiate the model
    #. specify parameters
    #. specify learning algorithm
    #. warm up
#. Train
#. Evaluate

Librairies
----------

.. code:: ipython3

    # Comment this if library is installed!
    import os
    import sys
    sys.path.insert(0, os.path.abspath(".."))

.. code:: ipython3

    from tqdm.notebook import tqdm_notebook
    from sklearn.metrics import classification_report
    
    import torch
    
    from datasets import load_dataset, Dataset, concatenate_datasets
    
    from transformers import AutoTokenizer
    from transformers.data.data_collator import DataCollatorWithPadding
    
    import esntorch.core.reservoir as res
    import esntorch.core.learning_algo as la
    import esntorch.core.pooling_strategy as ps
    import esntorch.core.esn as esn

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    # Set device (cpu or gpu if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device




.. parsed-literal::

    device(type='cuda')



Load and prepare data
---------------------

Load and tokenize data
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Custom functions for loading and preparing data
    def tokenize(sample):
        """Tokenize sample"""
        
        sample = tokenizer(sample['text'], truncation=True, padding=False, return_length=True)
        
        return sample
        
    def load_and_prepare_dataset(dataset_name, split, cache_dir):
        """
        Load dataset from the datasets library of HuggingFace.
        Tokenize and add length.
        """
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
        
        # Rename label column for tokenization purposes (use 'label-fine' for fine-grained labels)
        dataset = dataset.rename_column('label-coarse', 'labels')
        
        # Tokenize data
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column('length', 'lengths')
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'lengths'])
        
        return dataset

.. code:: ipython3

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and prepare data
    CACHE_DIR = 'cache_dir/' # put your path here
    
    full_dataset = load_and_prepare_dataset('trec', split=None, cache_dir=CACHE_DIR)
    train_dataset = full_dataset['train'].sort("lengths")
    test_dataset = full_dataset['test'].sort("lengths")
    
    # Create dict of all datasets
    dataset_d = {
        'train': train_dataset,
        'test': test_dataset
        }



.. code:: ipython3

    dataset_d




.. parsed-literal::

    {'train': Dataset({
         features: ['attention_mask', 'input_ids', 'label-fine', 'labels', 'lengths', 'text', 'token_type_ids'],
         num_rows: 5452
     }),
     'test': Dataset({
         features: ['attention_mask', 'input_ids', 'label-fine', 'labels', 'lengths', 'text', 'token_type_ids'],
         num_rows: 500
     })}



Create dataloaders
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Create dict of dataloaders
    
    dataloader_d = {}
    
    for k, v in dataset_d.items():
        dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, 
                                                      collate_fn=DataCollatorWithPadding(tokenizer))

.. code:: ipython3

    dataloader_d




.. parsed-literal::

    {'train': <torch.utils.data.dataloader.DataLoader at 0x7f6f03e56910>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7f6f03e56850>}



Instanciate the model
---------------------

Parameters
~~~~~~~~~~

For Deep ESNs, set the parameter ``deep``\ to ``True``, then customize
the number of layers (i.e., reservoirs) by specifying the parameter
``nb_layers``. Each of the other parameters (like ``dim``,
``distribution``, ``spectal radius``, etc.), can be specified in two
ways:

    #. List of values: in this case, the successive layers are built according to the successive values of the parameter in the list.
    #. Single value: in this case, the successive layers are all built according to the same value of the parameter. Please refer to the documentation for further details.

.. code:: ipython3

    # Deep ESN parameters
    esn_params = {
                'embedding': 'bert-base-uncased',
                'dim': [500, 400, 300],      # *** list of dims for the successive layers ***
                'sparsity': 0.9,
                'spectral_radius': 0.9,
                'leaking_rate': 0.5,
                'activation_function': 'tanh',
                'bias_scaling': 0.1,
                'input_scaling': 0.1,
                'mean': 0.0,
                'std': 1.0,     
                'learning_algo': None,       # initialzed below
                'criterion': None,           # initialzed below (only for learning algos trained with SGD)
                'optimizer': None,           # initialzed below (only for learning algos trained with SGD)
                'pooling_strategy': 'mean',
                'bidirectional': False,      # True, False
                'mode' : 'recurrent_layer',  # 'no_layer', 'linear_layer', 'recurrent_layer'
                'deep' : True,               # *** Deep ESN ***
                'nb_layers' : 3,             # *** 3 layers ***
                'device': device,
                'seed': 42
                }
    
    # Instantiate the ESN
    ESN = esn.EchoStateNetwork(**esn_params)


.. parsed-literal::

    Model downloaded: bert-base-uncased


Learning algorithm
~~~~~~~~~~~~~~~~~~

Choose your learning algo by un-commenting its associated cell.

The following algorithms are trained via a **direct method**. Hence,
there is no need to specify any *criterion* and *optimizer*:

    #. ``RidgeRegression`` (our implementation)
    #. ``RidgeRegression_skl`` (from scikit-learn)
    #. ``LinearSVC`` (from scikit-learn)
    #. ``LogisticRegression_skl`` (from scikit-learn)

.. code:: ipython3

    ESN.learning_algo = la.RidgeRegression(alpha=10.0)

.. code:: ipython3

    # ESN.learning_algo = la.RidgeRegression_skl(alpha=10.0)

.. code:: ipython3

    # ESN.learning_algo = la.LinearSVC(C=1.0)

.. code:: ipython3

    # ESN.learning_algo = la.LogisticRegression_skl()

The following algorithms are trained via a **gradient descent**.
Accordingly, a *criterion* and an *optimizer* must be specified:

    #. ``LogisticRegression`` (our implementation)
    #. ``DeepNN`` (our implementation)

.. code:: ipython3

    input_dim = sum([layer.dim for layer in ESN.layer.layers])
    
    if esn_params['bidirectional']:
        input_dim *= 2

.. code:: ipython3

    ESN.learning_algo = la.LogisticRegression(input_dim=input_dim, output_dim=6)

.. code:: ipython3

    # ESN.learning_algo = la.DeepNN([input_dim, 512, 256, 6])

.. code:: ipython3

    # Needs criterion and otpimizer
    
    ESN.criterion = torch.nn.CrossEntropyLoss()
    ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)

Warm up
~~~~~~~

.. code:: ipython3

    # Put model on device
    ESN = ESN.to(device)

.. code:: ipython3

    if isinstance(ESN.layer, res.LayerRecurrent):
        ESN.warm_up(dataset_d['train'].select(range(10)))

Train
-----

For **direct methods**, the parameters ``epochs`` and ``iter_steps`` are
ignored.

.. code:: ipython3

    ESN.fit(dataloader_d["train"], epochs=3, iter_steps=50)


.. parsed-literal::

    Performing gradient descent...
    Training complete.


Evaluate
--------

.. code:: ipython3

    # Train predictions and accuracy
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    train_acc

.. parsed-literal::

    75.51357300073367



.. code:: ipython3

    # Test predictions
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    test_acc

.. parsed-literal::

    81.0



.. code:: ipython3

    # Test classification report
    print(classification_report(test_pred.tolist(), 
                                dataset_d['test']['labels'].tolist(), 
                                digits=4))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0     0.8478    0.8478    0.8478       138
               1     0.4043    0.9500    0.5672        40
               2     0.3333    1.0000    0.5000         3
               3     0.9846    0.7442    0.8477        86
               4     0.9646    0.8450    0.9008       129
               5     0.9136    0.7115    0.8000       104
    
        accuracy                         0.8100       500
       macro avg     0.7414    0.8498    0.7439       500
    weighted avg     0.8766    0.8100    0.8270       500
    
