.. _installation:

============
Installation
============




Installation
------------

| This library is distributed on PyPi:
| https://pypi.org/project/esntorch/
| and can be installed with ``pip``, as shown below:

.. code::

    $ pip install esntorch

This command will automatically install the dependencies listed in ``requirements.txt``
together with the library itself.

Please visit the [installation page](docs/src/installation.rst) for more details.


GitHub
------

| The source code of the library is available on GitHub:
| https://github.com/PlaytikaResearch/EsnTorch
| It can be cloned with the following command:

.. code::

    $ git clone https://github.com/PlaytikaResearch/EsnTorch.git


Once cloned, you can install the library by running one of the following commands
from the root directory ``esntorch/``:

.. code::

    $ pip install .                        # install library + dependencies
    $ pip install -r requirements.txt      # install dependencies



More Info
---------

To create the HTML documentation run the following commands:

.. code::

    $ cd docs
    $ sphinx-apidoc -o source/ ../esntorch
    $ make clean
    $ make html



| To make the library pip instalable, create and ``.whl`` file and deploy it on PyPi:
| https://pypi.org/project/esntorch/
| Make sure that ``twine`` is installed and run the following commands:

.. code::

    $ python setup.py sdist bdist_wheel
    $ twine upload -r pypi dist/*



License
-------

[MIT License.](LICENSE)