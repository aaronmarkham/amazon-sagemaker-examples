Upgrade Your SageMaker Python SDK Notebooks
===========================================

Versions 2.0 and higher of the SageMaker Python SDK introduced some
changes that may require changes in your own code when upgrading. This
notebook serves as a helper for upgrading your code.

For more information, including what changes were made, see `the
documentation <https://sagemaker.readthedocs.io/en/stable/v2.html>`__.

Install the latest ``sagemaker`` version
----------------------------------------

Make sure you select the kernel that you normally use before pip
installing the latest version of ``sagemaker``:

.. code:: ipython3

    !pip install --upgrade "sagemaker>=2"

**IMPORTANT**: Now restart your kernel so that it picks up the updated
``sagemaker`` version.

Use the migration tool
----------------------

Upgrading ``sagemaker`` also installs a CLI tool to aid with updating
your code.

There are limitations with what the tool can handle:
https://sagemaker.readthedocs.io/en/stable/v2.html#limitations. In these
cases, you need to manually update your code.

Set the base path for where to look for notebooks. The code below points
to the base Jupyter home directory on a SageMaker Notebook Instance. If
you are using SageMaker Studio, uncomment the second line.

Modify the path as necessary so that it points to your notebooks.

.. code:: ipython3

    path = "/home/ec2-user/SageMaker"
    path = "/root"
    
    print(path)

Create a temporary directory for the upgraded files. This is just to err
on the safe side - you can also overwrite your existing files directly.

.. code:: ipython3

    import os
    
    output_path = os.path.join(path, "sagemaker-sdk-v2")
    !mkdir {output_path}

Now use the ``sagemaker-upgrade-v2`` tool to upgrade your notebooks. The
following code runs it for all notebooks found in the given path.

.. code:: ipython3

    import glob
    
    for file in glob.glob("**/*.ipynb", recursive=True):
        !sagemaker-upgrade-v2 --in-file {file} --out-file {output_path}/{file}

At this point, you may want to manually verify some notebooks and see
what changes were made.

When you are satisfied, you can copy them back to the original path:

.. code:: ipython3

    !cp -r {output_path}/* {path}/.

Cleanup
-------

Finally, remove the temporary directory created earlier.

.. code:: ipython3

    !rm -r {output_path}
