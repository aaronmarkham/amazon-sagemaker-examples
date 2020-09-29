Word-level language modeling using PyTorch
==========================================

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Data <#Data>`__
4. `Train <#Train>`__
5. `Host <#Host>`__

--------------

Background
----------

This example trains a multi-layer LSTM RNN model on a language modeling
task based on `PyTorch
example <https://github.com/pytorch/examples/tree/master/word_language_model>`__.
By default, the training script uses the Wikitext-2 dataset. We will
train a model on SageMaker, deploy it, and then use deployed model to
generate new text.

For more information about the PyTorch in SageMaker, please visit
`sagemaker-pytorch-containers <https://github.com/aws/sagemaker-pytorch-containers>`__
and
`sagemaker-python-sdk <https://github.com/aws/sagemaker-python-sdk>`__
github repositories.

--------------

Setup
-----

*This notebook was created and tested on an ml.p2.xlarge notebook
instance.*

Let’s start by creating a SageMaker session and specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See `the
   documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__
   for how to create these. Note, if more than one role is required for
   notebook instances, training, and/or hosting, please replace the
   sagemaker.get_execution_role() with appropriate full IAM role arn
   string(s).

.. code:: ipython2

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-rnn-lstm'
    
    role = sagemaker.get_execution_role()

Data
----

Getting the data
~~~~~~~~~~~~~~~~

As mentioned above we are going to use `the wikitext-2 raw
data <https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/>`__.
This data is from Wikipedia and is licensed CC-BY-SA-3.0. Before you use
this data for any other purpose than this example, you should understand
the data license, described at
https://creativecommons.org/licenses/by-sa/3.0/

.. code:: bash

    %%bash
    wget http://research.metamind.io.s3.amazonaws.com/wikitext/wikitext-2-raw-v1.zip
    unzip -n wikitext-2-raw-v1.zip
    cd wikitext-2-raw
    mv wiki.test.raw test && mv wiki.train.raw train && mv wiki.valid.raw valid


Let’s preview what data looks like.

.. code:: ipython2

    !head -5 wikitext-2-raw/train

Uploading the data to S3
~~~~~~~~~~~~~~~~~~~~~~~~

We are going to use the ``sagemaker.Session.upload_data`` function to
upload our datasets to an S3 location. The return value inputs
identifies the location – we will use later when we start the training
job.

.. code:: ipython2

    inputs = sagemaker_session.upload_data(path='wikitext-2-raw', bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))

Train
-----

Training script
~~~~~~~~~~~~~~~

We need to provide a training script that can run on the SageMaker
platform. The training script is very similar to a training script you
might run outside of SageMaker, but you can access useful properties
about the training environment through various environment variables,
such as:

-  ``SM_MODEL_DIR``: A string representing the path to the directory to
   write model artifacts to. These artifacts are uploaded to S3 for
   model hosting.
-  ``SM_OUTPUT_DATA_DIR``: A string representing the filesystem path to
   write output artifacts to. Output artifacts may include checkpoints,
   graphs, and other files to save, not including model artifacts. These
   artifacts are compressed and uploaded to S3 to the same S3 prefix as
   the model artifacts.

Supposing one input channel, ‘training’, was used in the call to the
PyTorch estimator’s ``fit()`` method, the following will be set,
following the format ``SM_CHANNEL_[channel_name]``:

-  ``SM_CHANNEL_TRAINING``: A string representing the path to the
   directory containing data in the ‘training’ channel.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
``model_dir`` so that it can be hosted later. Hyperparameters are passed
to your script as arguments and can be retrieved with an
``argparse.ArgumentParser`` instance.

In this notebook example, we will use Git integration. That is, you can
specify a training script that is stored in a GitHub, CodeCommit or
other Git repository as the entry point for the estimator, so that you
don’t have to download the scripts locally. If you do so, source
directory and dependencies should be in the same repo if they are
needed.

To use Git integration, pass a dict ``git_config`` as a parameter when
you create the ``PyTorch`` Estimator object. In the ``git_config``
parameter, you specify the fields ``repo``, ``branch`` and ``commit`` to
locate the specific repo you want to use. If authentication is required
to access the repo, you can specify fields ``2FA_enabled``,
``username``, ``password`` and token accordingly.

The script that we will use in this example is stored in GitHub repo
https://github.com/awslabs/amazon-sagemaker-examples/tree/training-scripts,
under the branch ``training-scripts``. It is a public repo so we don’t
need authentication to access it. Let’s specify the ``git_config``
argument here:

.. code:: ipython2

    git_config = {'repo': 'https://github.com/awslabs/amazon-sagemaker-examples.git', 'branch': 'training-scripts'}

Note that we do not specify ``commit`` in ``git_config`` here, in which
case the latest commit of the specified repo and branch will be used by
default.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
``model_dir`` so that it can be hosted later. Hyperparameters are passed
to your script as arguments and can be retrieved with an
``argparse.ArgumentParser`` instance.

For example, the script run by this notebook:
https://github.com/awslabs/amazon-sagemaker-examples/blob/training-scripts/pytorch-rnn-scripts/train.py.

For more information about training environment variables, please visit
`SageMaker Containers <https://github.com/aws/sagemaker-containers>`__.

In the current example we also need to provide source directory, because
training script imports data and model classes from other modules. The
source directory is
https://github.com/awslabs/amazon-sagemaker-examples/blob/training-scripts/pytorch-rnn-scripts/.
We should provide ‘pytorch-rnn-scripts’ for ``source_dir`` when creating
the Estimator object, which is a relative path inside the Git
repository.

Run training in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~

The PyTorch class allows us to run our training function as a training
job on SageMaker infrastructure. We need to configure it with our
training script and source directory, an IAM role, the number of
training instances, and the training instance type. In this case we will
run our training job on ``ml.p2.xlarge`` instance. As you can see in
this example you can also specify hyperparameters.

For this example, we’re specifying the number of epochs to be 1 for the
purposes of demonstration. We suggest at least 6 epochs for a more
meaningful result.

.. code:: ipython2

    from sagemaker.pytorch import PyTorch
    
    estimator = PyTorch(entry_point='train.py',
                        role=role,
                        framework_version='1.4.0',
                        train_instance_count=1,
                        train_instance_type='ml.p2.xlarge',
                        source_dir='pytorch-rnn-scripts',
                        git_config=git_config,
                        # available hyperparameters: emsize, nhid, nlayers, lr, clip, epochs, batch_size,
                        #                            bptt, dropout, tied, seed, log_interval
                        hyperparameters={
                            'epochs': 1,
                            'tied': True
                        })


After we’ve constructed our PyTorch object, we can fit it using the data
we uploaded to S3. SageMaker makes sure our data is available in the
local filesystem, so our training script can simply read the data from
disk.

.. code:: ipython2

    estimator.fit({'training': inputs})

Host
----

Hosting script
~~~~~~~~~~~~~~

We are going to provide custom implementation of ``model_fn``,
``input_fn``, ``output_fn`` and ``predict_fn`` hosting functions in a
separate file, which is in the same Git repo as the training script:
https://github.com/awslabs/amazon-sagemaker-examples/blob/training-scripts/pytorch-rnn-scripts/generate.py.
We will use Git integration for hosting too since the hosting code is
also in the Git repo.

You can also put your training and hosting code in the same file but you
would need to add a main guard (``if __name__=='__main__':``) for the
training code, so that the container does not inadvertently run it at
the wrong point in execution during hosting.

Import model into SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyTorch model uses a npy serializer and deserializer by default. For
this example, since we have a custom implementation of all the hosting
functions and plan on using JSON instead, we need a predictor that can
serialize and deserialize JSON.

.. code:: ipython2

    from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
    
    class JSONPredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)

Since hosting functions implemented outside of train script we can’t
just use estimator object to deploy the model. Instead we need to create
a PyTorchModel object using the latest training job to get the S3
location of the trained model data. Besides model data location in S3,
we also need to configure PyTorchModel with the script and source
directory (because our ``generate`` script requires model and data
classes from source directory), an IAM role.

.. code:: ipython2

    from sagemaker.pytorch import PyTorchModel
    
    training_job_name = estimator.latest_training_job.name
    desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
    trained_model_location = desc['ModelArtifacts']['S3ModelArtifacts']
    model = PyTorchModel(model_data=trained_model_location,
                         role=role,
                         framework_version='1.0.0',
                         entry_point='generate.py',
                         source_dir='pytorch-rnn-scripts',
                         git_config=git_config,
                         predictor_cls=JSONPredictor)

Create endpoint
~~~~~~~~~~~~~~~

Now the model is ready to be deployed at a SageMaker endpoint and we are
going to use the ``sagemaker.pytorch.model.PyTorchModel.deploy`` method
to do this. We can use a CPU-based instance for inference (in this case
an ml.m4.xlarge), even though we trained on GPU instances, because at
the end of training we moved model to cpu before returning it. This way
we can load trained model on any device and then move to GPU if CUDA is
available.

.. code:: ipython2

    predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Evaluate
~~~~~~~~

We are going to use our deployed model to generate text by providing
random seed, temperature (higher will increase diversity) and number of
words we would like to get.

.. code:: ipython2

    input = {
        'seed': 111,
        'temperature': 2.0,
        'words': 100
    }
    response = predictor.predict(input)
    print(response)

Cleanup
~~~~~~~

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython2

    sagemaker_session.delete_endpoint(predictor.endpoint)
