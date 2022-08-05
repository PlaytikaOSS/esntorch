Tutorial 1: Echo State Networks for Text Classification
=======================================================

This notebook presents a use case example of the ``EsnTorch`` library.
It describes the implementation of an **Echo State Network (ESN)** for
text classification on the **TREC-6** dataset (question classification).

The instantiation, training and evaluation of an ESN for text
classification is achieved via the following steps:

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

    {'train': <torch.utils.data.dataloader.DataLoader at 0x7fad4c231250>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7fad4c06cbb0>}



Instanciate the model
---------------------

Parameters
~~~~~~~~~~

Most parameters are self-explanatory for a reader familiar with ESNs.
Please refer to the documentation for further details.

The ``mode`` parameter represents the type of reservoir to be
considered:

    #. ``mode = recurrent_layer``: implements a **classical recurrent reservoir**, specified among others by its ``dim``, ``sparsity``, ``spectral_radius``, ``leaking_rate`` and ``activation_function``.
    #. ``mode = linear_layer``: implements a simple **linear layer** specified by its ``dim`` and ``activation_function``.
    #. ``mode = no_layer``: implements **the absence of reservoir**, meaning that the embedded inputs are directly fed to the the learning algorithms.

The comparison between the ``recurrent_layer`` and the
``no_layer``\ modes enables to assess the impact of the reservoir on the
results.

The comparison between the ``recurrent_layer`` and the
``linear_layer``\ modes allows to assess the importance of the
recurrence of the reservoir on the results.

.. code:: ipython3

    # ESN parameters
    esn_params = {
                'embedding': 'bert-base-uncased', # name of Hugging Face model
                'dim': 1000,
                'sparsity': 0.9,
                'spectral_radius': 0.9,
                'leaking_rate': 0.5,
                'activation_function': 'tanh', # 'tanh', 'relu'
                'bias_scaling': 0.1,
                'input_scaling': 0.1,
                'mean': 0.0,
                'std': 1.0,     
                'learning_algo': None,         # initialzed below
                'criterion': None,             # initialzed below (only for learning algos trained with SGD)
                'optimizer': None,             # initialzed below (only for learning algos trained with SGD)
                'pooling_strategy': 'mean',    # 'mean', 'last', None
                'bidirectional': False,        # True, False
                'mode' : 'recurrent_layer',    # 'no_layer', 'linear_layer', 'recurrent_layer'
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

    # if esn_params['mode'] == 'no_layer':
    #     input_dim = ESN.layer.input_dim
    # else:
    #     input_dim = ESN.layer.dim
    
    # if esn_params['bidirectional']:
    #     input_dim *= 2 

.. code:: ipython3

    # ESN.learning_algo = la.LogisticRegression(input_dim=input_dim, output_dim=6)

.. code:: ipython3

    # ESN.learning_algo = la.DeepNN([input_dim, 512, 256, 6])

.. code:: ipython3

    # # Needs criterion and otpimizer
    
    # ESN.criterion = torch.nn.CrossEntropyLoss()
    # ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)

Warm up
~~~~~~~

.. code:: ipython3

    # Put model on device
    ESN = ESN.to(device)

.. code:: ipython3

    # Warm up the ESN on multiple sentences
    if isinstance(ESN.layer, res.LayerRecurrent):
        ESN.warm_up(dataset_d['train'].select(range(10)))


Train
-----

For **direct methods**, the parameters ``epochs`` and ``iter_steps`` are
ignored.

.. code:: ipython3

    ESN.fit(dataloader_d["train"], epochs=3, iter_steps=50)


.. parsed-literal::

    Computing closed-form solution...
    Training complete.


Evaluate
--------

.. code:: ipython3

    # Train predictions and accuracy
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    train_acc

.. parsed-literal::

    93.01173881144534



.. code:: ipython3

    # Test predictions
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    test_acc

.. parsed-literal::

    93.8



.. code:: ipython3

    # Test classification report
    print(classification_report(test_pred.tolist(), 
                                dataset_d['test']['labels'].tolist(), 
                                digits=4))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0     0.9855    0.9189    0.9510       148
               1     0.8191    0.9506    0.8800        81
               2     0.7778    1.0000    0.8750         7
               3     0.9692    0.9844    0.9767        64
               4     0.9735    0.9402    0.9565       117
               5     0.9383    0.9157    0.9268        83
    
        accuracy                         0.9380       500
       macro avg     0.9106    0.9516    0.9277       500
    weighted avg     0.9429    0.9380    0.9390       500
    
