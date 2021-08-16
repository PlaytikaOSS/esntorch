# EsnTorch (v0.0.2)
**A library that implements echo state networks (ESNs) for natural language processing (NLP).**

EsnTorch is a library designed for the implementation of echo state networks (ESNs)
in the context of natural language processing (NLP). 
More specifically, EsnTorch allows to implement ESNs for text classification tasks.
EsnTorch is written in PyTorch. 

This library works for Python 3.6 and higher and PyTorch 1.7.1 and higher.


Installation
------------

This library is distributed on [PyPi](https://pypi.org/) and
can be installed with ``pip``. The latest release is version ``0.0.2``.
(DEPLOY TO PYPI AND UPDATE LINK ONCE AUTHORIZATION OBTAINED)

~~~~~~~~~~~~~~~~~~~~~~
$ pip install esntorch 
~~~~~~~~~~~~~~~~~~~~~~

The command above will automatically install all the dependencies listed in ``requirements.txt``. 

Please visit the [installation page](./docs/installation.rst) for more details.



Documentation
-------------
For more information, please read the full [documentation.](./docs/index.rst)



More Info
---------

The source code of the library is available on [GitHub](https://github.com/PlaytikaResearch/EsnTorch). 
It can be cloned via the following command:
 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git clone https://github.com/PlaytikaResearch/EsnTorch.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the library and the dependencies with one of the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install .                        # install library + dependencies
$ pip install -r requirements.txt      # install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the python wheel file ``pyabtest.whl`` for installation with ``pip`` run the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python setup.py sdist bdist_wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the HTML documentation run the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd docs
$ sphinx-apidoc -o source/ ../esntorch
$ make clean
$ make html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


License
-------

[Apache License, Version 2.0](LICENSE)