.. _installation:

============
Installation
============


Installation
------------

This library is distributed on `PyPi <https://pypi.org/>`_ and
can be installed with ``pip``. The latest release is version ``0.0.2``.
(DEPLOY TO PYPI AND UPDATE LINK ONCE AUTHORIZATION OBTAINED)

.. code::

    $ pip install esntorch

The command above will automatically install all the dependencies listed in ``requirements.txt``.


More Info
---------

The source code of the library is available on `GitHub <https://github.com/PlaytikaResearch/EsnTorch>`_.
It can be cloned via the following command:

.. code::

    $ git clone https://github.com/PlaytikaResearch/EsnTorch.git

You can install the library and the dependencies with one of the following commands:

.. code::

    $ pip install .                        # install library + dependencies
    $ pip install -r requirements.txt      # install dependencies

To create the python wheel file ``pyabtest.whl`` for installation with ``pip``
run the following command:

.. code::

    $ python setup.py sdist bdist_wheel

To create the HTML documentation run the following commands:

.. code::

    $ cd docs
    $ sphinx-apidoc -o source/ ../esntorch
    $ make clean
    $ make html


License
-------

`MIT License <https://opensource.org/licenses/MIT>`_

