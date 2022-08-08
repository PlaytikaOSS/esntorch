# EsnTorch (version 1.0.1)
**Echo state networks (ESNs) for natural language processing (NLP).**

``EsnTorch`` is a user-friendly python library designed for the implementation of **echo state networks (ESNs)**
in the context of **natural language processing (NLP)**, and more specifically, 
in the context of **text classification**.

``EsnTorch`` is written in ``PyTorch`` and requires Python 3.7 or higher.


Installation
------------

This library is distributed on [PyPi](https://pypi.org/project/esntorch/) and
can be installed with ``pip``, as shown below:

~~~~~~~~~~~~~~~~~~~~~~
$ pip install esntorch 
~~~~~~~~~~~~~~~~~~~~~~

This command will automatically install the dependencies listed in ``requirements.txt`` 
together with the library itself.

Please visit the [installation page](docs/src/installation.rst) for more details.


GitHub
------

The source code of the library is available on [GitHub](https://github.com/PlaytikaResearch/EsnTorch). 
It can be cloned with the following command:
 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git clone https://github.com/PlaytikaResearch/EsnTorch.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once cloned, you can install the library by running one of the following commands 
from the root directory ``esntorch/``:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install .                        # install library + dependencies
$ pip install -r requirements.txt      # install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Documentation
-------------

The [documentation page](https://playtikaresearch.github.io/esntorch/index.html) 
provides a detailed documentation of the library as well as tutorials covering 
its main functionalities.


More Info
---------

To create the HTML documentation run the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd docs
$ sphinx-apidoc -o source/ ../esntorch
$ make clean
$ make html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the python wheel file ``pyabtest.whl`` for installation with ``pip``, run the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python setup.py sdist bdist_wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


License
-------

[MIT License.](LICENSE)