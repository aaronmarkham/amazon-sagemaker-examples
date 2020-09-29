Visualizing Debugging Tensors of MXNet training
===============================================

Overview
~~~~~~~~

SageMaker Debugger is a new capability of Amazon SageMaker that allows
debugging machine learning models. It lets you go beyond just looking at
scalars like losses and accuracies during training and gives you full
visibility into all the tensors ‘flowing through the graph’ during
training. SageMaker Debugger helps you to monitor your training in near
real time using rules and would provide you alerts, once it has detected
an inconsistency in the training flow.

Using SageMaker Debugger is a two step process: Saving tensors and
Analysis. In this notebook we will run an MXNet training job and
configure SageMaker Debugger to store all tensors from this job.
Afterwards we will visualize those tensors in our notebook.

Dependencies
~~~~~~~~~~~~

Before we begin, let us install the library plotly if it is not already
present in the environment. If the below cell installs the library for
the first time, you’ll have to restart the kernel and come back to the
notebook. In addition to that, in order for our vizualiation to access
tensors let’s install smdebug - debugger library that provides API
access to tensors emitted during training job.

.. code:: ipython3

    ! python -m pip install plotly
    ! python -m pip install smdebug

Configure and run the training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we’ll call the Sagemaker MXNet Estimator to kick off a training job
with Debugger attached to it.

The ``entry_point_script`` points to the MXNet training script.

.. code:: ipython3

    import os
    import sagemaker
    from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
    from sagemaker.mxnet import MXNet
    
    sagemaker_session = sagemaker.Session()
    
    entry_point_script = './scripts/mxnet_gluon_save_all_demo.py'
    hyperparameters = {'batch-size': 256}
    base_job_name = 'mnist-tensor-plot'
    
    # Make sure to set this to your bucket and location
    BUCKET_NAME = sagemaker_session.default_bucket()
    LOCATION_IN_BUCKET = 'mnist-tensor-plot'
    s3_bucket_for_tensors = 's3://{BUCKET_NAME}/{LOCATION_IN_BUCKET}'.format(BUCKET_NAME=BUCKET_NAME, LOCATION_IN_BUCKET=LOCATION_IN_BUCKET)
    
    estimator = MXNet(role=sagemaker.get_execution_role(),
                      base_job_name=base_job_name,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      entry_point=entry_point_script,
                      framework_version='1.6.0',
                      train_max_run=3600,
                      sagemaker_session=sagemaker_session,
                      py_version='py3',
                      debugger_hook_config = DebuggerHookConfig(
                          s3_output_path=s3_bucket_for_tensors,  # Required
                          collection_configs=[
                              CollectionConfig(
                                  name="all_tensors",
                                  parameters={
                                      "include_regex": ".*",
                                      "save_steps": "1, 2, 3"
                                  }
                              )
                          ]
                      ))

Estimator described above will save all tensors of all layers during
steps 1, 2 and 3. Now, let’s start the training job:

.. code:: ipython3

    estimator.fit()

Get S3 location of tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can retrieve the S3 location of the tensors:

.. code:: ipython3

    tensors_path = estimator.latest_job_debugger_artifacts_path()
    print('S3 location of tensors is: ', tensors_path)

Download tensors from S3
~~~~~~~~~~~~~~~~~~~~~~~~

Next we download the tensors from S3, so that we can visualize them in
the notebook.

.. code:: ipython3

    folder_name = tensors_path.split("/")[-1]
    os.system("aws s3 cp --recursive " + tensors_path + " " + folder_name)
    print('Downloaded tensors into folder: ', folder_name)

Visualize
~~~~~~~~~

The main purpose of this class (TensorPlot) is to visualise the tensors
in your network. This could be to determine dead or saturated
activations, or the features maps the network.

To use this class (TensorPlot), you will need to supply the argument
regex with the tensors you are interested in. e.g., if you are
interested in activation outputs, then you need to supply the following
regex .\ *relu|.*\ tanh|.*sigmoid.

Another important argument is the ``sample_batch_id``, which allows you
to specify the index of the batch size to display. For example, given an
input tensor of size (batch_size, channel, width, height),
``sample_batch_id = n`` will display (n, channel, width, height). If you
set sample_batch_id = -1 then the tensors will be summed over the batch
dimension (i.e., ``np.sum(tensor, axis=0)``). If batch_sample_id is None
then each sample will be plotted as separate layer in the figure.

Here are some interesting use cases:

1) If you want to determine dead or saturated activations for instance
   ReLus that are always outputting zero, then you would want to sum the
   batch dimension (sample_batch_id=-1). The sum gives an indication
   which parts of the network are inactive across a batch.

2) If you are interested in the feature maps for the first image in the
   batch, then you should provide batch_sample_id=0. This can be helpful
   if your model is not performing well for certain set of samples and
   you want to understand which activations are leading to
   misprediction.

An example visualization of layer outputs: |image0|

``TensorPlot`` normalizes tensor values to the range 0 to 1 which means
colorscales are the same across layers. Blue indicates value close to 0
and yellow indicates values close to 1. This class has been designed to
plot convolutional networks that take 2D images as input and predict
classes or produce output images. You can use this for other types of
networks like RNNs, but you may have to adjust the class as it is
currently neglecting tensors that have more than 4 dimensions.

Let’s plot Relu output activations for the given MNIST training example.

.. |image0| image:: ./images/tensorplot.gif

.. code:: ipython3

    import tensor_plot 
    
    visualization = tensor_plot.TensorPlot(
        regex=".*relu_output", 
        path=folder_name,
        steps=10,  
        batch_sample_id=0,
        color_channel = 1,
        title="Relu outputs",
        label=".*sequential0_input_0",
        prediction=".*sequential0_output_0"
    )

If we plot too many layers, it can crash the notebook. If you encounter
performance or out of memory issues, then either try to reduce the
layers to plot by changing the ``regex`` or run this Notebook in
JupyterLab instead of Jupyter.

In the below cell we vizualize outputs of all layers, including final
classification. Please note that because training job ran only for 1
epoch classification accuracy is not high.

.. code:: ipython3

    visualization.fig.show(renderer="iframe")

For additional example of working with debugging tensors and visualizing
them in real time please feel free to try it out at `MXNet realtime
analysis <../mxnet_realtime_analysis/mxnet-realtime-analysis.ipynb>`__
example.
