Using SageMaker debugger to visualize class activation maps in CNNs
-------------------------------------------------------------------

This notebook will demonstrate how to use SageMaker debugger to plot
class activations maps for image classification models. A class
activation map (saliency map) is a heatmap that highlights the regions
in the image that lead the model to make a certain prediction. This is
especially useful:

1. if the model makes a misclassification and it is not clear why;

2. or to determine if the model takes all important features of an
   object into account

In this notebook we will train a
`ResNet <https://arxiv.org/abs/1512.03385>`__ model on the `German
Traffic Sign
Dataset <http://benchmark.ini.rub.de/?section=gtsrb&subsection=news>`__
and we will use SageMaker debugger to plot class activation maps in
real-time.

The following animation shows the saliency map for a particular traffic
sign as training progresses. Red highlights the regions with high
activation leading to the prediction, blue indicates low activation that
are less relevant for the prediction.

In the beginning the model will do a lot of mis-classifications as it
focuses on the wrong image regions e.g. the obstacle in the lower left
corner. As training progresses the focus shifts to the center of the
image, and the model becomes more and more confident in predicting the
class 3 (which is the correct class).

|image0|

There exist several methods to generate saliency maps e.g.
`CAM <http://cnnlocalization.csail.mit.edu/>`__,
`GradCAM <https://arxiv.org/abs/1610.02391>`__. The paper `Full-Gradient
Representation for Neural Network Visualization
[1] <https://arxiv.org/abs/1905.00780>`__ proposes a new method which
produces state of the art results. It requires intermediate features and
their biases. With SageMaker debugger we can easily retrieve these
tensors.

[1] *Full-Gradient Representation for Neural Network Visualization*:
Suraj Srinivas and Francois Fleuret, 2019, 1905.00780, arXiv

.. |image0| image:: images/example.gif

Customize the smdebug hook
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create saliency maps, the gradients of the prediction with respect to
the intermediate features need to be computed. To obtain this
information, we have to customize the `smdebug
hook <https://github.com/awslabs/sagemaker-debugger/blob/master/smdebug/pytorch/hook.py>`__.
The custom hook is defined in
`entry_point/custom_hook.py <entry_point/custom_hook.py>`__ During the
forward pass, we register a backward hook on the outputs. We also need
to get gradients of the input image, so we provide an additional
function that registers a backward hook on the input tensor.

The paper `Full-Gradient Representation for Neural Network Visualization
[1] <https://arxiv.org/abs/1905.00780>`__ distinguishes between implicit
and explicit biases. Implicit biases include running mean and variance
from BatchNorm layers. With SageMaker debugger we only get the explicit
biases which equals the beta paramater in the case of BatchNorm layers.
We extend the hook to also record running averages and variances for
BatchNorm layers.

.. code:: python

   import smdebug.pytorch as smd
          
   class CustomHook(smd.Hook):
       
       #register input image for backward pass, to get image gradients
       def image_gradients(self, image):
           image.register_hook(self.backward_hook("image"))
           
       def forward_hook(self, module, inputs, outputs):
           module_name = self.module_maps[module]   
           self._write_inputs(module_name, inputs)
           
           #register outputs for backward pass. this is expensive, so we will only do it during EVAL mode
           if self.mode == ModeKeys.EVAL:
               outputs.register_hook(self.backward_hook(module_name + "_output"))
               
               #record running mean and var of BatchNorm layers
               if isinstance(module, torch.nn.BatchNorm2d):
                   self._write_outputs(module_name + ".running_mean", module.running_mean)
                   self._write_outputs(module_name + ".running_var", module.running_var)
               
           self._write_outputs(module_name, outputs)
           self.last_saved_step = self.step

Replace in-place operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additionally we need to convert inplace operations, as they can
potentially overwrite values that are required to compute gradients. In
the case of PyTorch pre-trained ResNet model, ReLU activatons are per
default executed inplace. The following code sets ``inplace=False``

.. code:: ipython3

    def relu_inplace(model):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(model, child_name, torch.nn.ReLU(inplace=False))
            else:
                relu_inplace(child)

Download the dataset and upload it to Amazon S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we download the `German Traffic Sign
Dataset <http://benchmark.ini.rub.de/?section=gtsrb&subsection=news>`__
and upload it to Amazon S3. The training dataset consists of 43 image
classes.

.. code:: ipython3

    ! wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
    ! unzip -q GTSRB-Training_fixed.zip

The test dataset:

.. code:: ipython3

    ! wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
    ! unzip -q GTSRB_Final_Test_Images.zip

Now we upload the datasets to the SageMaker default bucket in Amazon S3.

.. code:: ipython3

    import boto3
    import sagemaker
    import os
    
    def upload_to_s3(path, directory_name, bucket, counter=-1):
        
        print("Upload files from" + path + " to " + bucket)
        client = boto3.client('s3')
        
        for path, subdirs, files in os.walk(path):
            path = path.replace("\\","/")
            print(path)
            for file in files[0:counter]:
                client.upload_file(os.path.join(path, file), bucket, directory_name+'/'+path.split("/")[-1]+'/'+file)
                
    boto_session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    bucket = sagemaker_session.default_bucket()
    
    upload_to_s3("GTSRB/Training", directory_name="train",  bucket=bucket)
    
    #we will compute saliency maps for all images in the test dataset, so we will only upload 4 images 
    upload_to_s3("GTSRB/Final_Test", directory_name="test", bucket=bucket, counter=4)

Before starting the SageMaker training job, we need to install some
libraries. We will use ``smdebug`` library to read, filter and analyze
raw tensors that are stored in Amazon S3. We will use ``opencv-python``
library to plot saliency maps as heatmap.

.. code:: ipython3

    ! pip install smdebug
    ! pip install opencv-python

SageMaker training
~~~~~~~~~~~~~~~~~~

Following code defines the SageMaker estimator. The entry point script
`train.py <entry_point/train.py>`__ defines the model training. It
downloads a pre-trained ResNet model and performs transfer learning on
the German traffic sign dataset.

Debugger hook configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we define a custom collection where we indicate regular expression
of tensor names to be included. Tensors from training phase are saved
every 100 steps, while tensors from validation phase are saved every
step. A step presents one forward and backward pass.

.. code:: ipython3

    from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
    
    debugger_hook_config = DebuggerHookConfig(
          collection_configs=[ 
              CollectionConfig(
                    name="custom_collection",
                    parameters={ "include_regex": ".*bn|.*bias|.*downsample|.*ResNet_input|.*image|.*fc_output|.*CrossEntropyLoss",
                                 "train.save_interval": "100",
                                 "eval.save_interval": "1" })])

Builtin rule
^^^^^^^^^^^^

In addition we run the training job with a builtin rule. We select here
the class imbalance rule that measures whether our training set is
imbalanced and/or whether the model has lower accurcay for certain
classes in the training dataset. The tensors that are passed into the
loss function ``CrossEntropyLoss`` are the labels and predictions. In
our example those tensors have the name ``CrossEntropyLoss_input_1`` and
``CrossEntropyLoss_input_0``. The rule uses those tensors to compute
class imbalance.

.. code:: ipython3

    from sagemaker.debugger import Rule, CollectionConfig, rule_configs
    
    class_imbalance_rule = Rule.sagemaker(base_config=rule_configs.class_imbalance(),
                                         rule_parameters={"labels_regex": "CrossEntropyLoss_input_1",
                                                          "predictions_regex": "CrossEntropyLoss_input_0",
                                                          "argmax":"True"})

SageMaker training
~~~~~~~~~~~~~~~~~~

Following code defines the SageMaker estimator. The entry point script
`train.py <entry_point/train.py>`__ defines the model training. It
downloads a pre-trained ResNet model and performs transfer learning on
the German traffic sign dataset.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    role = sagemaker.get_execution_role()
    
    pytorch_estimator = PyTorch(entry_point='train.py',
                                source_dir='entry_point',
                                role=role,
                                train_instance_type='ml.p2.xlarge',
                                train_instance_count=1,
                                framework_version='1.3.1',
                                py_version='py3',
                                hyperparameters = {'epochs': 10, 
                                                   'batch_size_train': 64,
                                                   'batch_size_val': 4,
                                                   'learning_rate': 0.001},
                               debugger_hook_config=debugger_hook_config,
                               rules=[class_imbalance_rule]
                               )

Now that we have defined the estimator we can call ``fit``, which will
start the training job on a ``ml.p3.2xlarge`` instance:

.. code:: ipython3

    pytorch_estimator.fit(inputs={'train': 's3://{}/train'.format(bucket), 
                                  'test': 's3://{}/test'.format(bucket)}, 
                          wait=False)

Check rule status
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    pytorch_estimator.latest_training_job.rule_job_summary()

Visualize saliency maps in real-time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the training job has started, SageMaker debugger will upload the
tensors of our model into S3. We can check the location in S3:

.. code:: ipython3

    path = pytorch_estimator.latest_job_debugger_artifacts_path()
    print('Tensors are stored in: {}'.format(path))

We can check the status of our training job, by executing
``describe_training_job``:

.. code:: ipython3

    job_name = pytorch_estimator.latest_training_job.name
    print('Training job name: {}'.format(job_name))
    
    client = pytorch_estimator.sagemaker_session.sagemaker_client
    
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
the trial:

.. code:: ipython3

    from smdebug.trials import create_trial
    
    trial = create_trial(path)

Now we can compute the saliency maps. The method described in
`Full-Gradient Representation for Neural Network Visualization
[1] <https://arxiv.org/abs/1905.00780>`__ requires all intermediate
features and their biases. The following cell retrieves the gradients
for the outputs of batchnorm and downsampling layers and the
corresponding biases. If you use a model other than ResNet you may need
to adjust the regular expressions in the following cell:

.. code:: ipython3

    biases, gradients = [], []
    
    for tname in trial.tensor_names(regex='.*gradient.*bn.*output|.*gradient.*downsample.1.*output'):
        gradients.append(tname)
        
    for tname in trial.tensor_names(regex='^(?=.*bias)(?:(?!fc).)*$'):
        biases.append(tname)

As mentioned in the beginning of the notebook, in the case of BatchNorm
layers, we need to compute the implicit biases. In the following code
cell we retrieve the necessary tensors:

.. code:: ipython3

    bn_weights, running_vars, running_means = [], [], []
    
    for tname in trial.tensor_names(regex='.*running_mean'):
        running_means.append(tname)
        
    for tname in trial.tensor_names(regex='.*running_var'):
        running_vars.append(tname)
    
    for tname in trial.tensor_names(regex='.*bn.*weight|.*downsample.1.*weight'):
        bn_weights.append(tname)  

We need to ensure that the tensors in the list are in order, e.g. bias
vector and gradients need to be for the same layer. Let’s have a look on
the tensors:

.. code:: ipython3

    for bias, gradient, weight, running_var, running_mean in zip(biases, gradients, bn_weights, running_vars, running_means):
        print(bias, gradient, weight, running_var, running_mean)

Here we define a helper function that is used later on to normalize
tensors:

.. code:: ipython3

    def normalize(tensor):
        tensor = tensor - np.min(tensor)
        tensor = tensor / np.max(tensor)
        return tensor

A helper function to plot saliency maps:

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    def plot(saliency_map, image, predicted_class, propability): 
        
        #clear matplotlib figure
        plt.clf()
        
        #revert normalization
        mean = [[[0.485]], [[0.456]], [[0.406]]]
        std = [[[0.229]], [[0.224]], [[0.225]]]
        image = image * std + mean
    
        #transpose image: color channel in last dimension
        image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8) 
        
        #create heatmap: we multiply it with -1 because we use
        #matplotlib to plot output results which inverts the colormap
        saliency_map = - saliency_map * 255
        saliency_map = saliency_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        
        #overlay original image with heatmap
        output_image = heatmap.astype(np.float32) + image.astype(np.float32)
        
        #normalize
        output_image = output_image / np.max(output_image)
        
        #plot
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))    
        ax0.imshow(image)
        ax1.imshow(output_image)
        ax0.set_axis_off()
        ax1.set_axis_off()
        ax0.set_title("Input image")
        ax1.set_title("Predicted class " + predicted_class + " with propability " + propability + "%")
        plt.show()   

A helper function to compute implicit biases:

.. code:: ipython3

    def compute_implicit_biases(bn_weights, running_vars, running_means, step):
        implicit_biases = []
        for weight_name, running_var_name, running_mean_name in zip(bn_weights, running_vars, running_means):
            weight = trial.tensor(weight_name).value(step_num=step, mode=modes.EVAL)
            running_var = trial.tensor(running_var_name).value(step_num=step, mode=modes.EVAL)
            running_mean = trial.tensor(running_mean_name).value(step_num=step, mode=modes.EVAL)
            implicit_biases.append(- running_mean / np.sqrt(running_var) * weight)
        return implicit_biases

Get available steps:

.. code:: ipython3

    import time
    steps = 0
    while steps == 0:
        steps = trial.steps()
        print('Waiting for tensors to become available...')
        time.sleep(3)
    print('\nDone')
    
    print('Getting tensors...')
    rendered_steps = []

We iterate over the tensors from the validation steps and compute the
saliency map for each item in the batch. To compute the saliency map, we
perform the following steps:

1. compute the implicit bias
2. multiply gradients and bias (sum of explicit and implicit bias)
3. normalize result
4. interpolate tensor to the input size of the original input image
5. create heatmap and overlay it with the original input image

.. code:: ipython3

    import numpy as np
    import cv2
    import scipy.ndimage
    import scipy.special
    from smdebug import modes
    from smdebug.core.modes import ModeKeys
    from smdebug.exceptions import TensorUnavailableForStep
    import os
    
    image_size = 224
    
    loaded_all_steps = False
    
    while not loaded_all_steps:
        
        # get available steps
        loaded_all_steps = trial.loaded_all_steps
        steps = trial.steps(mode=modes.EVAL)
        
        # quick way to get diff between two lists
        steps_to_render = list(set(steps).symmetric_difference(set(rendered_steps)))
    
        #iterate over available steps
        for step in sorted(steps_to_render):
            try:
    
                #get original input image
                image_batch = trial.tensor("ResNet_input_0").value(step_num=step, mode=modes.EVAL)
    
                #compute implicit biases from batchnorm layers
                implicit_biases = compute_implicit_biases(bn_weights, running_vars, running_means, step)
                
                for item in range(image_batch.shape[0]):
    
                    #input image
                    image = image_batch[item,:,:,:]
    
                    #get gradients of input image
                    image_gradient = trial.tensor("gradient/image").value(step_num=step, mode=modes.EVAL)[item,:]  
                    image_gradient = np.sum(normalize(np.abs(image_gradient * image)), axis=0)
                    saliency_map = image_gradient
    
                    for gradient_name, bias_name, implicit_bias in zip(gradients, biases, implicit_biases):
    
                        #get gradients and bias vectors for corresponding step
                        gradient = trial.tensor(gradient_name).value(step_num=step, mode=modes.EVAL)[item:item+1,:,:,:]
                        bias = trial.tensor(bias_name).value(step_num=step, mode=modes.EVAL) 
                        bias = bias + implicit_bias
    
                        #compute full gradient
                        bias = bias.reshape((1,bias.shape[0],1,1))
                        bias = np.broadcast_to(bias, gradient.shape)
                        bias_gradient = normalize(np.abs(bias * gradient))
    
                        #interpolate to original image size
                        for channel in range(bias_gradient.shape[1]):
                            interpolated = scipy.ndimage.zoom(bias_gradient[0,channel,:,:], image_size/bias_gradient.shape[2], order=1)
                            saliency_map += interpolated 
    
    
                    #normalize
                    saliency_map = normalize(saliency_map) 
                    
                    #predicted class and propability
                    predicted_class = trial.tensor("fc_output_0").value(step_num=step, mode=modes.EVAL)[item,:]
                    print("Predicted class:", np.argmax(predicted_class))
                    scores = np.exp(np.asarray(predicted_class))
                    scores = scores / scores.sum(0)
                    
                    #plot image and heatmap
                    plot(saliency_map, image, str(np.argmax(predicted_class)), str(int(np.max(scores) * 100)) )
                    
            except TensorUnavailableForStep:
                print("Tensor unavailable for step {}".format(step))
                
        rendered_steps.extend(steps_to_render)
        
        time.sleep(5)
        
    print('\nDone')

