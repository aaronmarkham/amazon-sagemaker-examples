Tensor analysis using Amazon SageMaker Debugger
===============================================

Looking at the distributions of activation inputs/outputs, gradients and
weights per layer can give useful insights. For instance, it helps to
understand whether the model runs into problems like neuron saturation,
whether there are layers in your model that are not learning at all or
whether the network consists of too many layers etc.

The following animation shows the distribution of gradients of a
convolutional layer from an example application as the training
progresses. We can see that it starts as Gaussian distribution but then
becomes more and more narrow. We can also see that the range of
gradients starts very small (order of :math:`1e-5`) and becomes even
tinier as training progresses. If tiny gradients are observed from the
start of training, it is an indication that we should check the
hyperparameters of our model.

|image0|

In this notebook we will train a poorly configured neural network and
use Amazon SageMaker Debugger with custom rules to aggregate and analyse
specific tensors. Before we proceed let us install the smdebug binary
which allows us to perform interactive analysis in this notebook. After
installing it, please restart the kernel, and when you come back skip
this cell.

Installing smdebug
~~~~~~~~~~~~~~~~~~

.. |image0| image:: images/example.gif

.. code:: ipython3

    !  python -m pip install smdebug

Configuring the inputs for the training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we’ll call the Sagemaker MXNet Estimator to kick off a training job
. The ``entry_point_script`` points to the MXNet training script. The
users can create a custom *SessionHook* in their training script. If
they chose not to create such hook in the training script (similar to
the one we will be using in this example) Amazon SageMaker Debugger will
create the appropriate *SessionHook* based on specified
*DebugHookConfig* parameters.

The ``hyperparameters`` are the parameters that will be passed to the
training script. We choose ``Uniform(1)`` as initializer and learning
rate of ``0.001``. This leads to the model not training well because the
model is poorly initialized.

The goal of a good intialization is - to break the symmetry such that
parameters do not receive same gradients and updates - to keep variance
similar across layers

A bad intialization may lead to vanishing or exploiding gradients and
the model not training at all. Once the training is running we will look
at the distirbutions of activation inputs/outputs, gradients and weights
across the training to see how these hyperparameters influenced the
training.

.. code:: ipython3

    entry_point_script = 'mnist.py'
    bad_hyperparameters = {'initializer': 2, 'lr': 0.001}

.. code:: ipython3

    import sagemaker
    from sagemaker.mxnet import MXNet
    from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
    import boto3
    import os
    
    estimator = MXNet(role=sagemaker.get_execution_role(),
                      base_job_name='mxnet',
                      train_instance_count=1,
                      train_instance_type='ml.m5.xlarge',
                      train_volume_size=400,
                      source_dir='src',
                      entry_point=entry_point_script,
                      hyperparameters=bad_hyperparameters,
                      framework_version='1.6.0',
                      py_version='py3',
                      debugger_hook_config = DebuggerHookConfig(
                          collection_configs=[
                            CollectionConfig(
                                name="all",
                                parameters={
                                    "include_regex": ".*",
                                    "save_interval": "100"
                                }
                            )
                         ]
                       )
                    )

Start the training job

.. code:: ipython3

    estimator.fit(wait=False)

Get S3 location of tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get information related to the training job:

.. code:: ipython3

    job_name = estimator.latest_training_job.name
    client = estimator.sagemaker_session.sagemaker_client
    description = client.describe_training_job(TrainingJobName=job_name)
    description

We can retrieve the S3 location of the tensors:

.. code:: ipython3

    path = estimator.latest_job_debugger_artifacts_path()
    print('Tensors are stored in: ', path)

We can check the status of our training job, by executing
``describe_training_job``:

.. code:: ipython3

    job_name = estimator.latest_training_job.name
    print('Training job name: {}'.format(job_name))
    
    client = estimator.sagemaker_session.sagemaker_client
    
    description = client.describe_training_job(TrainingJobName=job_name)

We can access the tensors from S3 once the training job is in status
``Training`` or ``Completed``. In the following code cell we check the
job status:

.. code:: ipython3

    import time
    
    if description['TrainingJobStatus'] != 'Completed':
        while description['SecondaryStatus'] not in {'Training', 'Completed'}:
            description = client.describe_training_job(TrainingJobName=job_name)
            primary_status = description['TrainingJobStatus']
            secondary_status = description['SecondaryStatus']
            print('Current job status: [PrimaryStatus: {}, SecondaryStatus: {}]'.format(primary_status, secondary_status))
            time.sleep(15)

Once the job is in status ``Training`` or ``Completed``, we can create
the trial that allows us to access the tensors in Amazon S3.

.. code:: ipython3

    from smdebug.trials import create_trial
    
    trial1 = create_trial(path)

We can check the available steps. A step presents one forward and
backward pass.

.. code:: ipython3

    trial1.steps()

As training progresses more steps will become available.

Next we will access specific tensors like weights, gradients and
activation outputs and plot their distributions. We will use Amazon
SageMaker Debugger and define custom rules to retrieve certain tensors.
Rules are supposed to return True or False. However in this notebook we
will use custom rules to store dictionaries of aggregated tensors per
layer and step, which we then plot afterwards.

A custom rule inherits from the smdebug Rule class and implements the
function ``invoke_at_step``. This function is called everytime tensors
of a new step become available:

::


   from smdebug.rules.rule import Rule

   class MyCustomRule(Rule):
       def __init__(self, base_trial):
           super().__init__(base_trial)
           
       def invoke_at_step(self, step):  
           if np.max(self.base_trial.tensor('conv0_relu_output_0').value(step) < 0.001:
               return True
       return False

Above example rule checks if the first convolutional layer outputs only
small values. If so the rule returns ``True`` which corresponds to an
``Issue found``, otherwise False ``No Issue found``.

Activation outputs
~~~~~~~~~~~~~~~~~~

This rule will use Amazon SageMaker Debugger to retrieve tensors from
the ReLU output layers. It sums the activations across batch and steps.
If there is a large fraction of ReLUs outputing 0 across many steps it
means that the neuron is dying.

.. code:: ipython3

    from smdebug.trials import create_trial
    from smdebug.rules.rule_invoker import invoke_rule
    from smdebug.exceptions import NoMoreData
    from smdebug.rules.rule import Rule
    import numpy as np
    import utils
    import collections
    import os
    from IPython.display import Image

.. code:: ipython3

    class ActivationOutputs(Rule):
        def __init__(self, base_trial):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict() 
        
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*relu_output'):
                if "gradients" not in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = collections.OrderedDict()
                        if step not in self.tensors[tname]:
                            self.tensors[tname][step] = 0
                        neg_values = np.where(tensor <= 0)[0]
                        if len(neg_values) > 0:
                            self.logger.info(f" Step {step} tensor  {tname}  has {len(neg_values)/tensor.size*100}% activation outputs which are smaller than 0 ")
                        batch_over_sum = np.sum(tensor, axis=0)/tensor.shape[0]
                        self.tensors[tname][step] += batch_over_sum
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = ActivationOutputs(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/activation_outputs.gif')

.. code:: ipython3

    Image(url='images/activation_outputs.gif')

Activation Inputs
~~~~~~~~~~~~~~~~~

In this rule we look at the inputs into activation function, rather than
the output. This can be helpful to understand if there are extreme
negative or positive values that saturate the activation functions.

.. code:: ipython3

    class ActivationInputs(Rule):
        def __init__(self, base_trial):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict() 
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*relu_input'):
                if "gradients" not in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}
                        if step not in self.tensors[tname]:
                            self.tensors[tname][step] = 0
                        neg_values = np.where(tensor <= 0)[0]
                        if len(neg_values) > 0:
                            self.logger.info(f" Tensor  {tname}  has {len(neg_values)/tensor.size*100}% activation inputs which are smaller than 0 ")
                        batch_over_sum = np.sum(tensor, axis=0)/tensor.shape[0]
                        self.tensors[tname][step] += batch_over_sum
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = ActivationInputs(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/activation_inputs.gif')

We can see that second convolutional layer ``conv1_relu_input_0``
receives only negative input values, which means that all ReLUs in this
layer output 0.

.. code:: ipython3

    Image(url='images/activation_inputs.gif')

Gradients
~~~~~~~~~

The following code retrieves the gradients and plots their distribution.
If variance is tiny, that means that the model parameters do not get
updated effectively with each training step or that the training has
converged to a minimum.

.. code:: ipython3

    class GradientsLayer(Rule):
        def __init__(self, base_trial):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict()  
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*gradient'):
                try:
                    tensor = self.base_trial.tensor(tname).value(step)
                    if tname not in self.tensors:
                        self.tensors[tname] = {}
    
                    self.logger.info(f" Tensor  {tname}  has gradients range: {np.min(tensor)} {np.max(tensor)} ")
                    self.tensors[tname][step] = tensor
                except:
                    self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = GradientsLayer(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')

Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/gradients.gif')

.. code:: ipython3

    Image(url='images/gradients.gif')

Check variance across layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rule retrieves gradients, but this time we compare variance of
gradient distribution across layers. We want to identify if there is a
large difference between the min and max variance per training step. For
instance, very deep neural networks may suffer from vanishing gradients
the deeper we go. By checking this ratio we can determine if we run into
such a situation.

.. code:: ipython3

    class GradientsAcrossLayers(Rule):
        def __init__(self, base_trial, ):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict()  
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*gradient'):
                try:
                    tensor = self.base_trial.tensor(tname).value(step)
                    if step not in self.tensors:
                        self.tensors[step] = [np.inf, 0]
                    variance = np.var(tensor.flatten())
                    if variance < self.tensors[step][0]:
                        self.tensors[step][0] = variance
                    elif variance > self.tensors[step][1]:
                        self.tensors[step][1] = variance             
                    self.logger.info(f" Step {step} current ratio: {self.tensors[step][0]} {self.tensors[step][1]} Ratio: {self.tensors[step][1] / self.tensors[step][0]}") 
                except:
                    self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = GradientsAcrossLayers(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')

Let’s check min and max values of the gradients across layers:

.. code:: ipython3

    for step in rule.tensors:
        print("Step", step, "variance of gradients: ", rule.tensors[step][0], " to ",  rule.tensors[step][1])

Distribution of weights
~~~~~~~~~~~~~~~~~~~~~~~

This rule retrieves the weight tensors and checks the variance. If the
distribution does not change much across steps it may indicate that the
learning rate is too low, that gradients are too small or that the
training has converged to a minimum.

.. code:: ipython3

    class WeightRatio(Rule):
        def __init__(self, base_trial, ):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict()  
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*weight'):
                if "gradient" not in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}
                     
                        self.logger.info(f" Tensor  {tname}  has weights with variance: {np.var(tensor.flatten())} ")
                        self.tensors[tname][step] = tensor
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = WeightRatio(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/weights.gif')

.. code:: ipython3

    Image(url='images/weights.gif')

Inputs
~~~~~~

This rule retrieves layer inputs excluding activation inputs.

.. code:: ipython3

    class Inputs(Rule):
        def __init__(self, base_trial, ):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict()  
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*input'):
                if "relu" not in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}
                     
                        self.logger.info(f" Tensor  {tname}  has inputs with variance: {np.var(tensor.flatten())} ")
                        self.tensors[tname][step] = tensor
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = Inputs(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/layer_inputs.gif')

.. code:: ipython3

    Image(url='images/layer_inputs.gif')

Layer outputs
~~~~~~~~~~~~~

This rule retrieves outputs of layers excluding activation outputs.

.. code:: ipython3

    class Outputs(Rule):
        def __init__(self, base_trial, ):
            super().__init__(base_trial)  
            self.tensors = collections.OrderedDict() 
            
        def invoke_at_step(self, step):
            for tname in self.base_trial.tensor_names(regex='.*output'):
                if "relu" not in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}
                     
                        self.logger.info(f" Tensor  {tname}  has inputs with variance: {np.var(tensor.flatten())} ")
                        self.tensors[tname][step] = tensor
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
            return False
    
    rule = Outputs(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(rule.tensors, filename='images/layer_outputs.gif')

.. code:: ipython3

    Image(url='images/layer_outputs.gif')

Comparison
~~~~~~~~~~

In the previous section we have looked at the distribution of gradients,
activation outputs and weights of a model that has not trained well due
to poor initialization. Now we will compare some of these distributions
with a model that has been well intialized.

.. code:: ipython3

    entry_point_script = 'mnist.py'
    hyperparameters = {'lr': 0.01}

.. code:: ipython3

    estimator = MXNet(role=sagemaker.get_execution_role(),
                      base_job_name='mxnet',
                      train_instance_count=1,
                      train_instance_type='ml.m5.xlarge',
                      train_volume_size=400,
                      source_dir='src',
                      entry_point=entry_point_script,
                      hyperparameters=hyperparameters,
                      framework_version='1.6.0',
                      py_version='py3',
                      debugger_hook_config = DebuggerHookConfig(
                          collection_configs=[
                            CollectionConfig(
                                name="all",
                                parameters={
                                    "include_regex": ".*",
                                    "save_interval": "100"
                                }
                            )
                         ]
                       )
                    )
                      

Start the training job

.. code:: ipython3

    estimator.fit(wait=False)

Get S3 path where tensors have been stored

.. code:: ipython3

    path = estimator.latest_job_debugger_artifacts_path()
    print('Tensors are stored in: ', path)

Check the status of the training job:

.. code:: ipython3

    job_name = estimator.latest_training_job.name
    print('Training job name: {}'.format(job_name))
    
    client = estimator.sagemaker_session.sagemaker_client
    
    description = client.describe_training_job(TrainingJobName=job_name)
    
    if description['TrainingJobStatus'] != 'Completed':
        while description['SecondaryStatus'] not in {'Training', 'Completed'}:
            description = client.describe_training_job(TrainingJobName=job_name)
            primary_status = description['TrainingJobStatus']
            secondary_status = description['SecondaryStatus']
            print('Current job status: [PrimaryStatus: {}, SecondaryStatus: {}]'.format(primary_status, secondary_status))
            time.sleep(15)

Now we create a new trial object ``trial2``:

.. code:: ipython3

    from smdebug.trials import create_trial
    
    trial2 = create_trial(path)

Gradients
^^^^^^^^^

Lets compare distribution of gradients of the convolutional layers of
both trials. ``trial`` is the trial object of the first training job,
``trial2`` is the trial object of second training job. We can now easily
compare tensors from both training jobs.

.. code:: ipython3

    rule = GradientsLayer(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


.. code:: ipython3

    dict_gradients = {}
    dict_gradients['gradient/conv0_weight_bad_hyperparameters'] = rule.tensors['gradient/conv0_weight']
    dict_gradients['gradient/conv1_weight_bad_hyperparameters'] = rule.tensors['gradient/conv1_weight']

Second trial:

.. code:: ipython3

    rule = GradientsLayer(trial2)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


.. code:: ipython3

    dict_gradients['gradient/conv0_weight_good_hyperparameters'] = rule.tensors['gradient/conv0_weight']
    dict_gradients['gradient/conv1_weight_good_hyperparameters'] = rule.tensors['gradient/conv1_weight']

Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(dict_gradients, filename='images/gradients_comparison.gif')

In the case of the poorly initalized model, gradients are fluctuating a
lot leading to very high variance.

.. code:: ipython3

    Image(url='images/gradients_comparison.gif')

Activation inputs
^^^^^^^^^^^^^^^^^

Lets compare distribution of activation inputs of both trials.

.. code:: ipython3

    rule = ActivationInputs(trial1)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


.. code:: ipython3

    dict_activation_inputs = {}
    dict_activation_inputs['conv0_relu_input_0_bad_hyperparameters'] = rule.tensors['conv0_relu_input_0']
    dict_activation_inputs['conv1_relu_input_0_bad_hyperparameters'] = rule.tensors['conv1_relu_input_0']

Second trial

.. code:: ipython3

    rule = ActivationInputs(trial2)
    try:
        invoke_rule(rule)
    except NoMoreData:
        print('The training has ended and there is no more data to be analyzed. This is expected behavior.')


.. code:: ipython3

    dict_activation_inputs['conv0_relu_input_0_good_hyperparameters'] = rule.tensors['conv0_relu_input_0']
    dict_activation_inputs['conv1_relu_input_0_good_hyperparameters'] = rule.tensors['conv1_relu_input_0']

Plot the histograms

.. code:: ipython3

    utils.create_interactive_matplotlib_histogram(dict_activation_inputs, filename='images/activation_inputs_comparison.gif')

The distribution of activation inputs into first activation layer
``conv0_relu_input_0`` look quite similar in both trials. However in the
case of the second layer they drastically differ.

.. code:: ipython3

    Image(url='images/activation_inputs_comparison.gif')
