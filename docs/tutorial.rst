.. _tutorial:

Tutorial
========

This notebook provides a use case example of the ``EsnTorch`` library.
It described the implementation of an Echo State Network (ESN) for text
classification on the TREC-6 dataset.

The instantiation, training and evaluation of an ESN for text
classification is achieved via the following steps:
#. Import the required modules
#. Create the dataloaders
#. Instantiate the ESN by specifying:
* a reservoir
* a loss function
* a learning algorithm
#. Train the ESN
#. Results

Librairies
----------

.. code:: ipython3

    #!pip install datasets==1.7.0

.. code:: ipython3

    import os
    import sys
    sys.path.insert(0, os.path.abspath(".."))
    # sys.path.insert(0, os.path.abspath("../.."))

.. code:: ipython3

    # import numpy as np
    from sklearn.metrics import classification_report
    
    import torch
    
    from datasets import load_dataset, Dataset, concatenate_datasets
    
    from transformers import AutoTokenizer
    from transformers.data.data_collator import DataCollatorWithPadding
    
    import esntorch.core.reservoir as res
    import esntorch.core.learning_algo as la
    import esntorch.core.merging_strategy as ms
    import esntorch.core.esn as esn


Device and Seed
---------------

.. code:: ipython3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device




.. parsed-literal::

    device(type='cpu')




Load and Tokenize Data
----------------------

.. code:: ipython3

    # Custom functions for loading and preparing data
    
    def tokenize(sample):
        """Tokenize sample: variable 'tokenizer' contains the """
        sample = tokenizer(sample['text'], truncation=True, padding=False)
        return sample
        
    def add_lengths(sample):
        """Add 'lengths' field to sort batch by length"""
        sample["lengths"] = sum(sample["input_ids"] != 0)
        return sample
        
    def load_and_prepare_dataset(dataset_name, split, cache_dir):
        """
        Load dataset from the datasets library of HuggingFace.
        Tokenize and add length.
        """
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
        
        # Rename label column (use 'label-fine' for fine-grained labels)
        # Used for tokenization purposes.
        dataset = dataset.rename_column('label-coarse', 'labels')
        
        # Tokenize data
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Add 'lengths' feature
        dataset = dataset.map(add_lengths, batched=False)
        
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


.. parsed-literal::

    Using custom data configuration default


.. parsed-literal::

    Downloading and preparing dataset trec/default (download: 350.79 KiB, generated: 403.39 KiB, post-processed: Unknown size, total: 754.18 KiB) to cache_dir/trec/default/1.1.0/751da1ab101b8d297a3d6e9c79ee9b0173ff94c4497b75677b59b61d5467a9b9...



.. parsed-literal::

    Downloading:   0%|          | 0.00/336k [00:00<?, ?B/s]



.. parsed-literal::

    Downloading:   0%|          | 0.00/23.4k [00:00<?, ?B/s]



.. parsed-literal::

    0 examples [00:00, ? examples/s]



.. parsed-literal::

    0 examples [00:00, ? examples/s]


.. parsed-literal::

    Dataset trec downloaded and prepared to cache_dir/trec/default/1.1.0/751da1ab101b8d297a3d6e9c79ee9b0173ff94c4497b75677b59b61d5467a9b9. Subsequent calls will reuse this data.



.. parsed-literal::

      0%|          | 0/6 [00:00<?, ?ba/s]



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?ba/s]



.. parsed-literal::

      0%|          | 0/5452 [00:00<?, ?ex/s]



.. parsed-literal::

      0%|          | 0/500 [00:00<?, ?ex/s]


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



.. code:: ipython3

    # Create dict of dataloaders
    
    dataloader_d = {}
    
    for k, v in dataset_d.items():
        dataloader_d[k] = torch.utils.data.DataLoader(v, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer))

.. code:: ipython3

    dataloader_d




.. parsed-literal::

    {'train': <torch.utils.data.dataloader.DataLoader at 0x7f7f799db090>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7f7f99333ad0>}




Model
-----

.. code:: ipython3

    # ESN parameters
    esn_params = {
                'embedding_weights': 'bert-base-uncased', # TEXT.vocab.vectors,
                'distribution' : 'uniform',               # uniform, gaussian
                'input_dim' : 768,                        # dim of BERT encoding!
                'reservoir_dim' : 500,
                'bias_scaling' : 1.0,
                'sparsity' : 0.99,
                'spectral_radius' : 0.9,
                'leaking_rate': 0.5,
                'activation_function' : 'tanh',
                'input_scaling' : 1.0,
                'mean' : 0.0,
                'std' : 1.0,
                #'learning_algo' : None, # initialzed below
                #'criterion' : None,     # initialzed below
                #'optimizer' : None,     # initialzed below
                'merging_strategy' : 'mean',
                'bidirectional' : False, # True
                'device' : device,
                'seed' : 42
                 }
    
    # Instantiate the ESN
    ESN = esn.EchoStateNetwork(**esn_params)
    
    # Define the learning algo of the ESN
    ESN.learning_algo = la.RidgeRegression(alpha=10)
    
    # Put the ESN on the device (CPU or GPU)
    ESN = ESN.to(device)

.. code:: ipython3

    # Warm up the ESN on 3 sentences
    nb_sentences = 3
    
    for i in range(nb_sentences): 
        sentence = dataset_d["train"].select([i])
        dataloader_tmp = torch.utils.data.DataLoader(sentence, 
                                                     batch_size=1, 
                                                     collate_fn=DataCollatorWithPadding(tokenizer))  
    
        for sentence in dataloader_tmp:
            ESN.warm_up(sentence)


Training
--------

.. code:: ipython3

    # training the ESN
    ESN.fit(dataloader_d["train"])


Results
-------

.. code:: ipython3

    # Train predictions and accuracy
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    train_acc.item()




.. parsed-literal::

    86.86720275878906



.. code:: ipython3

    # Test predictions and accuracy
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    test_acc.item()




.. parsed-literal::

    87.80000305175781



.. code:: ipython3

    # Test classification report
    print(classification_report(test_pred.tolist(), dataset_d['test']['labels'].tolist()))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       0.94      0.88      0.91       148
               1       0.65      0.86      0.74        71
               2       0.56      1.00      0.71         5
               3       0.95      0.89      0.92        70
               4       0.95      0.90      0.92       119
               5       0.91      0.85      0.88        87
    
        accuracy                           0.88       500
       macro avg       0.83      0.90      0.85       500
    weighted avg       0.89      0.88      0.88       500
    


