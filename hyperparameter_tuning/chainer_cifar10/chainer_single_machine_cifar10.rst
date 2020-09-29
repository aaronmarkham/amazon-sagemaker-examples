Hyperparameter Tuning with Chainer
----------------------------------

`VGG <https://arxiv.org/pdf/1409.1556v6.pdf>`__ is an architecture for
deep convolution networks. In this example, we use convolutional
networks to perform image classification using the CIFAR-10 dataset.
CIFAR-10 consists of 60000 32x32 colour images in 10 classes, with 6000
images per class. There are 50000 training images and 10000 test images.

We’ll use SageMaker’s hyperparameter tuning to train multiple
convolutional networks, experimenting with different hyperparameter
combinations. After that, we’ll find the model with the best
performance, deploy it to Amazon SageMaker hosting, and then classify
images using the deployed model.

This notebook uses the Chainer script and estimator setup from `the
“Training with Chainer”
notebook <files/chainer_single_machine_cifar10.ipynb>`__.

.. code:: ipython3

    # Setup
    from sagemaker import get_execution_role
    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    
    # This role retrieves the SageMaker-compatible role used by this notebook instance.
    role = get_execution_role()

Downloading training and test data
----------------------------------

We use helper functions provided by ``chainer`` to download and
preprocess the CIFAR10 data.

.. code:: ipython3

    import chainer
    
    from chainer.datasets import get_cifar10
    
    train, test = get_cifar10()

Uploading the data
------------------

We save the preprocessed data to the local filesystem, and then use the
``sagemaker.Session.upload_data`` function to upload our datasets to an
S3 location. The return value ``inputs`` identifies the S3 location,
which we will use when we start the Training Job.

.. code:: ipython3

    import os
    import shutil
    
    import numpy as np
    
    train_data = [element[0] for element in train]
    train_labels = [element[1] for element in train]
    
    test_data = [element[0] for element in test]
    test_labels = [element[1] for element in test]
    
    
    try:
        os.makedirs('/tmp/data/train_cifar')
        os.makedirs('/tmp/data/test_cifar')
        np.savez('/tmp/data/train_cifar/train.npz', data=train_data, labels=train_labels)
        np.savez('/tmp/data/test_cifar/test.npz', data=test_data, labels=test_labels)
        train_input = sagemaker_session.upload_data(
                          path=os.path.join('/tmp', 'data', 'train_cifar'),
                          key_prefix='notebook/chainer_cifar/train')
        test_input = sagemaker_session.upload_data(
                          path=os.path.join('/tmp', 'data', 'test_cifar'),
                          key_prefix='notebook/chainer_cifar/test')
    finally:
        shutil.rmtree('/tmp/data')
    print('training data at %s' % train_input)
    print('test data at %s' % test_input)

Writing the Chainer script
--------------------------

We use a single script to train and host a Chainer model. The training
part is similar to a script you might run outside of SageMaker.

The hosting part requires implementing certain functions. Here, we’ve
defined only ``model_fn()``, which loads model artifacts that were
created during training. The other functions will take on default values
as described
`here <https://github.com/aws/sagemaker-python-sdk#model-serving>`__.

For a more in-depth discussion of this script see `the “Training with
Chainer” notebook <files/chainer_single_machine_cifar10.ipynb>`__.

For more on writing Chainer scripts to run on SageMaker, or for more on
the Chainer container itself, please see the following repositories:

-  For writing Chainer scripts to run on SageMaker:
   https://github.com/aws/sagemaker-python-sdk
-  For more on the Chainer container and default hosting functions:
   https://github.com/aws/sagemaker-chainer-containers

.. code:: ipython3

    !pygmentize 'src/chainer_cifar_vgg_single_machine.py'

Running hyperparameter tuning jobs on SageMaker
-----------------------------------------------

To specify options for a training job using Chainer, we construct a
``Chainer`` estimator using the
`sagemaker-python-sdk <https://github.com/aws/sagemaker-python-sdk>`__.
We pass in an ``entry_point``, the name of a script that contains a
couple of functions with certain signatures (``train()`` and
``model_fn()``), and a ``source_dir``, a directory containing all code
to run inside the Chainer container. This script will be run on
SageMaker in a container that invokes these functions to train and load
Chainer models.

For this example, we’re specifying the number of epochs to be 1 for the
purposes of demonstration. We suggest at least 50 epochs for a more
meaningful result.

.. code:: ipython3

    from sagemaker.chainer.estimator import Chainer
    
    chainer_estimator = Chainer(entry_point='chainer_cifar_vgg_single_machine.py',
                                source_dir="src",
                                role=role,
                                sagemaker_session=sagemaker_session,
                                train_instance_count=1,
                                train_instance_type='ml.p2.xlarge',
                                hyperparameters={'epochs': 1, 'batch-size': 64})

We then need to pass this estimator to a ``HyperparameterTuner``. For
the ``HyperparameterTuner`` class, we define the following options for
running hyperparameter tuning jobs: \* ``hyperparameter_ranges``: the
hyperparameters we’d like to tune and their possible values. We have
three different types of hyperparameters that can be tuned: categorical,
continuous, and integer. \* ``objective_metric_name``: the objective
metric we’d like to tune. \* ``metric_definitions``: the name of the
objective metric as well as the regular expression (regex) used to
extract the metric from the CloudWatch logs of each training job. \*
``max_jobs``: number of training jobs to run in total. \*
``max_parallel_jobs``: number of training jobs to run simultaneously.

For this example, we are going to tune on learning rate. In general, if
possible, it’s best to specify a value as the least restrictive type, so
we define learning rate as a continuous parameter ranging between 0.5
and 0.6 rather than, say, a categorical parameter with possible values
of 0.5, 0.55, and 0.6.

.. code:: ipython3

    from sagemaker.tuner import ContinuousParameter
    
    hyperparameter_ranges = {'learning-rate': ContinuousParameter(0.05, 0.06)}

Next, we define our objective metric, which we use to evaluate each
training job. This consists of a name and a regex. The training script
in this example uses Chainer’s
```PrintReport`` <https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.PrintReport.html>`__
to print out metrics for each epoch, which looks something like this
when run for 50 epochs (truncated here):

::

   epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
   #033[J1           2.33857     1.86438               0.175811       0.254479                  47.5526
   #033[J2           1.78559     1.59937               0.298095       0.376493                  79.5099
   #033[J3           1.50956     1.38693               0.422015       0.469646                  111.372
   ...
   #033[J48          0.378797    0.573417              0.879842       0.821955                  1548.58
   #033[J49          0.373226    0.573498              0.879516       0.812201                  1580.56
   #033[J50          0.369154    0.485158              0.882242       0.843451                  1612.49

The regex we use captures the fourth number in the last row, which is
the validation accuracy for the final epoch in the training job. Because
we’re using only one epoch for demonstration purposes, our regex has
‘J1’ in it, but the ‘1’ should be replaced with the number of epochs
used for each training job.

.. code:: ipython3

    objective_metric_name = 'Validation-accuracy'
    metric_definitions = [{'Name': 'Validation-accuracy', 'Regex': '\[J1\s+\d\.\d+\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)'}]

Finally, we need to define how many training jobs to run. We recommend
you set the parallel jobs value to less than 10% of the total number of
training jobs, but we are setting it higher here to keep this example
short. We are also setting ``max_jobs`` to a low value to shorten the
time needed for the hyperparameter tuning job to complete, but note that
running only two jobs won’t demonstrate any meaningful hyperparameter
tuning results.

.. code:: ipython3

    max_jobs = 2
    max_parallel_jobs = 2

.. code:: ipython3

    from sagemaker.tuner import HyperparameterTuner
    
    chainer_tuner = HyperparameterTuner(estimator=chainer_estimator,
                                        objective_metric_name=objective_metric_name,
                                        hyperparameter_ranges=hyperparameter_ranges,
                                        metric_definitions=metric_definitions,
                                        max_jobs=max_jobs,
                                        max_parallel_jobs=max_parallel_jobs)

With our tuner, we can now invoke ``fit()`` to start a hyperparameter
tuning job:

.. code:: ipython3

    chainer_tuner.fit({'train': train_input, 'test': test_input})

Waiting for the hyperparameter tuning job to complete
-----------------------------------------------------

Now we wait for the hyperparameter tuning job to complete. We have a
convenience method, ``wait()``, that will block until the hyperparameter
tuning job has completed. We can call that here to see if the
hyperparameter tuning job is still running; the cell will finish running
when the hyperparameter tuning job has completed.

.. code:: ipython3

    chainer_tuner.wait()

Deploying the Trained Model
---------------------------

After training, we use the tuner object to create and deploy a hosted
prediction endpoint with the best training job. We can use a CPU-based
instance for inference (in this case an ``ml.m4.xlarge``), even though
we trained on GPU instances.

The predictor object returned by ``deploy()`` lets us call the new
endpoint and perform inference on our sample images using the model from
the best training job found during hyperparameter tuning.

.. code:: ipython3

    predictor = chainer_tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

CIFAR10 sample images
~~~~~~~~~~~~~~~~~~~~~

We’ll use these CIFAR10 sample images to test the service:

Predicting using SageMaker Endpoint
-----------------------------------

We batch the images together into a single NumPy array to obtain
multiple inferences with a single prediction request.

.. code:: ipython3

    from skimage import io
    import numpy as np
    
    def read_image(filename):
        img = io.imread(filename)
        img = np.array(img).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        img *= 1. / 255.
        img = img.reshape(3, 32, 32)
        return img
    
    
    def read_images(filenames):
        return np.array([read_image(f) for f in filenames])
    
    filenames = ['images/airplane1.png',
                 'images/automobile1.png',
                 'images/bird1.png',
                 'images/cat1.png',
                 'images/deer1.png',
                 'images/dog1.png',
                 'images/frog1.png',
                 'images/horse1.png',
                 'images/ship1.png',
                 'images/truck1.png']
    
    image_data = read_images(filenames)

The predictor runs inference on our input data and returns a list of
predictions whose argmax gives the predicted label of the input data.

.. code:: ipython3

    response = predictor.predict(image_data)
    
    for i, prediction in enumerate(response):
        print('image {}: prediction: {}'.format(i, prediction.argmax(axis=0)))

Cleanup
-------

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython3

    chainer_tuner.delete_endpoint()
