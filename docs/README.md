Generate documentation via Sphynx
---------------------------------

1. The``.rst`` files are those from which the documentation is built:
   1. The ``.rst`` files corresponding to the documentation pages written by myself are in ``docs/src``.
   2. The ``.rst`` files used by Sphynx to generate the code documentation are in ``docs/src/source``.
2. Run the following commands to automatically build the doc:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd docs/src/
$ sphinx-apidoc -o source/ ../../esntorch
$ make clean
$ make html
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3. Copy all files in the folder `/src/build/html/` directly into `docs/`. MAke ure you keep the `/docs/src/` folder.
4. Delete the created folder `/src/build/`.


Deploy documentation on GitHub
------------------------------

1. Go to the GitHub page of the library.
2. Go into the `Settings` and then `Pages`.
3. In the `Source`, add the `/docs` folder such that the following holds:

    **Your GitHub Pages site is currently being built from the `/docs` folder in the main branch**

Deploy documentation on Artifactory
-----------------------------------

1. Compress the folder `docs/build/html/` to `docs.zip`
2. Deploy `docs.zip` to Artifactory (drag and drop). The file will be unzipped automatically.