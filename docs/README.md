To generate doc automatically via Sphynx:

:code::

    $ cd docs/src/
    $ sphinx-apidoc -o source/ ../../esntorch
    $ make clean
    $ make html

To deploy doc to Artifactory:

1. Compress the folder `docs/build/html/` to `docs.zip`
2. Deploy `docs.zip` to Artifactory (drag and drop). The file will be unzipped automatically.