### To generate the documentation automatically via Sphynx:

1. Run the following commands to build the doc:


    $ cd docs/src/
    $ sphinx-apidoc -o source/ ../../esntorch
    $ make clean
    $ make html

2. Then copy all the files in the folder `/src/build/html/` directly into `/src/` 
and finally delete the created folder `/src/build/`.

### To deploy the documentation on GitHub:

1. Go to the GitHub page of the library
2. Go into the `Settings` and then `Pages`
3. In the `Source`, add the `/docs` folder such that the following holds:

    **Your GitHub Pages site is currently being built from the `/docs` folder in the main branch**

### To deploy the documentation to Artifactory:

1. Compress the folder `docs/build/html/` to `docs.zip`
2. Deploy `docs.zip` to Artifactory (drag and drop). The file will be unzipped automatically.