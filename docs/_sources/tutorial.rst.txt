Tutorial
========

This notebook provides a use case example of the ``EsnTorch`` library.
It described the implementation of an **Echo State Network (ESN)** for
text classification on the **TREC-6** dataset.

The instantiation, training and evaluation of an ESN for text
classification is achieved via the following steps: - Import the
required modules - Create the dataloaders - Instantiate the ESN by
specifying: - a reservoir - a loss function - a learning algorithm -
Train the ESN - Training and testing results

Librairies
----------

.. code:: ipython3

    # !pip install transformers==4.8.2
    # !pip install datasets==1.7.0

.. code:: ipython3

    # Comment this if library is installed
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
    import esntorch.core.merging_strategy as ps
    import esntorch.core.esn as esn

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2


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


.. parsed-literal::

    Using custom data configuration default
    Reusing dataset trec (cache_dir/trec/default/1.1.0/751da1ab101b8d297a3d6e9c79ee9b0173ff94c4497b75677b59b61d5467a9b9)



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=6.0), HTML(value='')))


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


.. parsed-literal::

    


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

    {'train': <torch.utils.data.dataloader.DataLoader at 0x7fa0a176b8d0>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7fa0a176ba10>}




Model
-----

.. code:: ipython3

    # ESN parameters
    esn_params = {
                'embedding_weights': 'bert-base-uncased', # TEXT.vocab.vectors,
                'distribution' : 'uniform',               # uniform, gaussian
                'input_dim' : 768,                        # dim of BERT encoding!
                'reservoir_dim' : 1000,
                'bias_scaling' : 0., #1.0742377381236705, # 1.0,
                'sparsity' : 0.,
                'spectral_radius' : 0.7094538192983408, # 0.9,
                'leaking_rate': 0.17647315261153904, # 0.5,
                'activation_function' : 'relu',
                'input_scaling' : 0.1, # 1.0,
                'mean' : 0.0,
                'std' : 1.0,
                #'learning_algo' : None,     # initialzed below
                #'criterion' : None,         # initialzed below
                #'optimizer' : None,         # initialzed below
                'merging_strategy' : 'mean',
                'bidirectional' : False,     # True
                'device' : device,
                'mode' : 'esn',              # 'no_layer', 'linear_layer'
                'seed' : 42345
                 }
    
    # Instantiate the ESN
    ESN = esn.EchoStateNetwork(**esn_params)
    
    # Define the learning algo of the ESN
    # Ridge Regression
    ESN.learning_algo = la.RidgeRegression(alpha=7.843536845714804)
    
    # Logistic Regression (uncomment below)
    # if esn_params['mode'] == 'no_layer':
    #     input_dim = esn_params['input_dim']
    # else:
    #     input_dim = esn_params['reservoir_dim']
        
    # ESN.learning_algo = la.LogisticRegression(input_dim=input_dim, output_dim=6)
    # ESN.criterion = torch.nn.CrossEntropyLoss()                                  # loss
    # ESN.optimizer = torch.optim.Adam(ESN.learning_algo.parameters(), lr=0.01)    # optimizer
    
    # Put the ESN on the device (CPU or GPU)
    ESN = ESN.to(device)


.. parsed-literal::

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


.. parsed-literal::

    Model downloaded: bert-base-uncased


.. code:: ipython3

    # Warm up the ESN on multiple sentences
    nb_sentences = 10
    
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

    %%time
    # training the ESN
    ESN.fit(dataloader_d["train"])


.. parsed-literal::

    CPU times: user 2min 34s, sys: 7.61 s, total: 2min 41s
    Wall time: 2min 15s


.. code:: ipython3

    # %%time
    # # training the ESN (Logistic Regression, gradient descent)
    # ESN.fit(dataloader_d["train"], epochs=10, iter_steps=10)


Results
-------

.. code:: ipython3

    # Train predictions and accuracy
    train_pred, train_acc = ESN.predict(dataloader_d["train"], verbose=False)
    train_acc.item()




.. parsed-literal::

    92.33309173583984



.. code:: ipython3

    # Test predictions and accuracy
    test_pred, test_acc = ESN.predict(dataloader_d["test"], verbose=False)
    test_acc.item()




.. parsed-literal::

    93.4000015258789



.. code:: ipython3

    # Test classification report
    print(classification_report(test_pred.tolist(), 
                                dataset_d['test']['labels'].tolist(), 
                                digits=4))


.. parsed-literal::

                  precision    recall  f1-score   support
    
               0     0.9638    0.9048    0.9333       147
               1     0.8085    0.9620    0.8786        79
               2     0.6667    1.0000    0.8000         6
               3     0.9692    0.9265    0.9474        68
               4     0.9823    0.9407    0.9610       118
               5     0.9630    0.9512    0.9571        82
    
        accuracy                         0.9340       500
       macro avg     0.8922    0.9475    0.9129       500
    weighted avg     0.9407    0.9340    0.9354       500
    

