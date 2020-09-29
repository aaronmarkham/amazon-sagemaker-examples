Reduce Training Time with Apache MXNet and Horovod on Amazon SageMaker
======================================================================

Amazon SageMaker is a fully managed service that provides every
developer and data scientist with the ability to build, train, and
deploy machine learning (ML) models quickly. SageMaker removes the heavy
lifting from each step of the machine learning process to make it easier
to develop high quality models. As datasets continue to increase in
size, additional compute is required to reduce the amount of time it
takes to train. One method to scale horizontally and add these
additional resources on SageMaker is through the use of Horovod and
Apache MXNet. In this post, we will show how users can reduce training
time with MXNet and Horovod on SageMaker. Finally, we will demonstrate
how you can improve performance even more with advanced sections on
Horovod Timeline, Horovod Autotune, Horovod Fusion, and MXNet
Optimization.

Distributed Training
====================

Distributed training of neural networks for computer vision (CV) and
natural language processing (NLP) applications has become ubiquitous.
With Apache MXNet, you only need to modify a few lines of code to enable
distributed training. Distributed training allows you to reduce training
time by scaling horizontally. The goal is to split training tasks into
independent subtasks and execute these across multiple devices. There
are primarily two approaches for training in parallel: \* Data
parallelism: You distribute the data and share the model across multiple
compute resources. \* Model parallelism: You distribute the model and
share transformed data across multiple compute resources.

In this blog, we focus on data parallelism. Specifically, we discuss how
Horovod and MXNet allow you to train efficiently on SageMaker.

Horovod Overview
================

Horovod is an open-source distributed deep learning framework. It
leverages efficient inter-GPU and inter-node communication methods such
as NVIDIA Collective Communications Library (NCCL) and Message Passing
Interface (MPI) to distribute and aggregate model parameters between
workers. Horovod makes distributed deep learning fast and easy by
utilizing a single-GPU training script and scaling it across many GPUs
in parallel. It is built on top of the ring-allreduce communication
protocol. This approach allows each training process (i.e. process
running on a single GPU device) to talk to its peers and exchange
gradients by averaging (“reduction”) on a subset of gradients. The
diagram below illustrates how ring-allreduce works.

.. raw:: html

   <center>

 Fig. 1 The ring-allreduce algorithm allows worker nodes to average
gradients and disperse them to all nodes without the need for a
parameter server (source)

.. raw:: html

   </center>

Apache MXNet is integrated with Horovod through the distributed training
APIs defined in Horovod and you can convert the non-distributed training
by following the higher level code skeleton, which will also be shown
below. Although this greatly simplifies the process of using Horovod,
other complexities need to be considered. For example, you may need to
install additional software and libraries to resolve your
incompatibilities for making distributed training work. Horovod requires
a certain version of Open MPI, and if you want to leverage
high-performance training on NVIDIA GPUs you need to install NCCL
libraries. These complexities are amplified when you scale across
multiple devices, since you need to make sure all the software and
libraries in the new nodes are properly installed and configured. Amazon
SageMaker includes all the required libraries to run distributed
training with MXNet and Horovod. Prebuilt Sagemaker Docker Images come
with popular open-source deep learning frameworks and pre-configured
CUDA, cuDNN, MPI, and NCCL libraries. SageMaker manages the difficult
process of properly installing and configuring your cluster. Together
SageMaker and MXNet simplify training with Horovod by managing the
complexities to support distributed training at scale.

Test Problem and Dataset
========================

| In order to benchmark the efficiencies realized by Horovod we trained
  the notoriously resource-intensive model architecture Mask-RCNN. This
  model architecture was first introduced in 2018, and is currently
  considered the baseline model architectures for a popular Computer
  Vision task, Instance Segmentation (Mask-RCNN). Mask-RCNN builds upon
  Faster-RCNN by adding a mask for segmentation. Apache MXNet provides
  pre-built Mask-RCNN model as part of the GluonCV Model Zoo,
  simplifying the process of training these models. To train our object
  detection and instance segmentation models, we used the popular
  COCO2017 dataset. This dataset provides more than 200,000 images and
  their corresponding labels. COCO2017 dataset is considered an industry
  standard for benchmarking CV models.
| GluonCV is a computer-vision toolkit built on top of MXNet. It
  provides out-of-the-box support for various CV tasks including data
  loading and preprocessing for many common algorithm’s available within
  its model zoo. It also has a tutorial on how to get the COCO2017
  dataset. In order to make this process replicable for Amazon SageMaker
  users, we will show an entire end-to-end process for training
  Mask-RCNN with Horovod and MXNet. To begin, we first open the Jupyter
  environment on your Sagemaker Notebook and use the conda_mxnet_p36
  kernel. Next, we install the required Python packages:

.. code:: ipython3

    !pip install gluoncv==0.8.0b20200723 -q
    !pip install pycocotools -q

.. code:: ipython3

    import mxnet as mx
    #import gluoncv as gcv
    import os
    import sagemaker
    import subprocess
    from sagemaker.mxnet.estimator import MXNet
    
    sagemaker_session = sagemaker.Session() # can use LocalSession() to run container locally
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()

.. code:: ipython3

    #We will use GluonCV's tool to download our data
    gcv.utils.download('https://gluon-cv.mxnet.io/_downloads/b6ade342998e03f5eaa0f129ad5eee80/mscoco.py',path='./')

.. code:: ipython3

    #Now to install the dataset. Warning, this may take a while
    !python mscoco.py --download-dir data

.. code:: ipython3

    bucket_name = #INSERT BUCKET NAME

.. code:: ipython3

    #Upload the dataset to your s3 bucket
    !aws s3 cp './data/' s3://<INSERT BUCKET NAME>/ --recursive --quiet

Here is the standard way of performing training via paramater servers.

.. code:: ipython3

    # Define basic configuration of your Sagemaker Parameter/Horovod cluster.
    num_instances = 1 #How many nodes you want to use
    gpu_per_instance = 8 #How many gpus are on this instance
    bs = 1 # Batch-Size per gpu
    
    #Parameter Server variation
    hyperparameters = {
        'epochs':12, 'batch-size': bs, 'horovod':'false','lr':.01,'amp':'true',
        'val-interval':6,'num-workers':16}
    
    for instance_family in ['ml.p3.16xlarge','ml.p3dn.24xlarge']:#Which instance you want to use
        estimator = MXNet(
            entry_point='train_mask_rcnn.py',
            source_dir='./source',
            role=role,
            train_max_run=72*60*60,
            train_instance_type=instance_family,
            train_instance_count=num_instances,
            framework_version='1.6.0',
            train_volume_size=100,
            base_job_name =s.split('_')[1] + 'rcnn-' + str(num_instances)+ '-' + '-'.join(instance_family.split('.')[1:]),
            py_version='py3',
            hyperparameters=hyperparameters
        )
    
        estimator.fit(
            {'data':'s3://' + bucket_name + '/data'},
            wait=False
        )

| The Amazon SageMaker MXNet Estimator Class supports Horovod via the
  “distributions” parameter. We need to add a predefined “mpi” parameter
  with the “enabled” flag, and define the following additional
  parameters:
| \* processes_per_host (int): Number of processes MPI should launch on
  each host. This parameter is usually equal to number of GPU devices
  available on any given instance. \* custom_mpi_options (str): Any
  custom mpirun flags passed in this field are added to the mpirun
  command and executed by Amazon SageMaker for Horovod training.

Here is an example of how to initialize the distributions parameters:

.. code:: ipython3

    # Define basic configuration of your Sagemaker Parameter/Horovod cluster.
    num_instances = 1 #How many nodes you want to use
    gpu_per_instance = 8 #How many gpus are on this instance
    bs = 1 # Batch-Size per gpu
    
    distributions = {'mpi': {
                        'enabled': True,
                        'processes_per_host': gpu_per_instance,
                            }
                    }
    
    hyperparameters = {
        'epochs':12, 'batch-size':bs, 'horovod':'true','lr':.01,'amp':'true',
        'val-interval':6,'num-workers':15}
    
    for num_instances in [1,3]:
        for instance_family in ['ml.p3.16xlarge','ml.p3dn.24xlarge']:#Which instance you want to use
            estimator = MXNet(
                entry_point='train_mask_rcnn.py',
                source_dir='./source',
                role=role,
                train_max_run=72*60*60,
                train_instance_type=instance_family,
                train_instance_count=num_instances,
                framework_version='1.6.0',
                train_volume_size=100,
                base_job_name =s.split('_')[1] + 'rcnn-hvd-bs-' + str(num_instances)+ '-' + '-'.join(instance_family.split('.')[1:]),
                py_version='py3',
                hyperparameters=hyperparameters,
                distributions=distributions
            )
    
            estimator.fit(
                {'data':'s3://' + bucket_name + '/data'},
                wait=False
            )

Training Script with Horovod Support
====================================

In order to use Horovod in your training script, only a few
modifications are required. Code samples and instructions are available
in the `Horovod
documentation <https://horovod.readthedocs.io/en/stable/mxnet.html>`__.
In addition, many GluonCV models in the model zoo have scripts which
already support Horovod out of the box. Let’s review the key changes
that are required for Horovod to correctly work on Amazon SageMaker with
Apache MXNet. The following code follows directly from `Horovod’s
documentation <https://horovod.readthedocs.io/en/stable/mxnet.html>`__.

::

   import mxnet as mx
   import horovod.mxnet as hvd
   from mxnet import autograd

   # Initialize Horovod, this has to be done first as it activates Horovod.
   hvd.init()

   # GPU setup 
   context =[mx.gpu(hvd.local_rank())] #local_rank is the specific gpu on that 
   # instance
   num_gpus = hvd.size() #This is how many total GPUs you will be using.

   #Typically, in your data loader you will want to shard your dataset. For 
   # example, in the train_mask_rcnn.py script 
   train_sampler = \
           gcv.nn.sampler.SplitSortedBucketSampler(...,
                                                   num_parts=hvd.size() if args.horovod else 1,
                                                   part_index=hvd.rank() if args.horovod else 0)

   #Normally, we would shard the dataset first for Horovod.
   val_loader = mx.gluon.data.DataLoader(dataset, len(ctx), ...) #... is for your # other arguments

       
   # You build and initialize your model as usual.
   model = ...

   # Fetch and broadcast the parameters.
   params = model.collect_params()
   if params is not None:
       hvd.broadcast_parameters(params, root_rank=0)

   # Create DistributedTrainer, a subclass of gluon.Trainer.
   trainer = hvd.DistributedTrainer(params, opt)

   # Create loss function and train your model as usual. 

Results
=======

We trained Faster-RCNN and Mask-RCNN with similar parameters, except
batch-size and learning rate, on the COCO 2017 dataset to provide
training performance and accuracy benchmarks.

.. raw:: html

   <center>

 Fig. 2 Horovod Training Results.

.. raw:: html

   </center>

We used the approach for scaling our batch-size and learning rate from
the `“Accurate, Large Minibatch SGD: Training ImageNet in 1
Hour” <https://arxiv.org/abs/1706.02677>`__ paper. With the improvement
in training time enabled by Horovod and SageMaker, Scientists can focus
more on improving their algorithms instead of waiting for jobs to finish
training. Using Horovod Scientists can train in parallel across multiple
instances with marginal impact to mean Average Precision (mAP).

Optimizing Horovod Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Horovod provides several additional utilities which allow you to analyze
and optimize training performance. Horovod Autotune Finding the optimal
combinations of parameters for a given combination of model and cluster
size may require several iterations of trial-and-error. The Autotune
feature allows you to automate this trial-and-error activity within a
single training job and uses Bayesian optimization to search through the
parameter space for the most performant combination of parameters.
Horovod will search for the best combination of parameters in the first
cycles of a training job, and the once best combination is defined,
Horovod will write the best configuration in the Autotune log and use
this combination for the remainder of the training job. See more details
`here <https://horovod.readthedocs.io/en/stable/autotune.html>`__.

To enable Autotune and capture the search log, pass the following
parameters in your MPI configuration:

::

   {
       'mpi':
       {
           'enabled': True,
           'custom_mpi_options': '-x HOROVOD_AUTOTUNE=1 -x HOROVOD_AUTOTUNE_LOG=/opt/ml/output/autotune_log.csv'
       }
   }

Horovod Timeline
~~~~~~~~~~~~~~~~

Horovod Timeline is a report available after training completion which
captures all activities in the Horovod ring. This is useful to
understand which operations are taking the longest time and will
identify optimization opportunities. Refer to `this
article <https://horovod.readthedocs.io/en/stable/timeline.html>`__ for
more details. for more details. To generate a Timeline file, add the
following parameters in your MPI command:

::

   {
       'mpi':
       {
           'enabled': True,
           'custom_mpi_options': '-x HOROVOD_TIMELINE=/opt/ml/output/timeline.json'
       }
   }

Note, that ``/opt/ml/output`` is a directory with specific purpose.
After training job completion, Amazon Sagemaker automatically archives
all files in this directory and uploads it to S3 location defined by
user. That’s where your Timeline report will be available for your
further analysis.

Note, that /opt/ml/output is a directory with a specific purpose. After
training job completion, Amazon Sagemaker automatically archives all
files in this directory and uploads it to an Amazon S3 location defined
by the user in the Python SageMaker SDK API.

Tensor Fusion
~~~~~~~~~~~~~

The Tensor Fusion feature allows users to perform batch **allreduce**
operations at training time. This typically results in better overall
performance, see additional details
`here <https://horovod.readthedocs.io/en/stable/tensor-fusion.html>`__.
By default, Tensor Fusion is enabled and has a buffer size of 64MB. You
can modify buffer size using a custom MPI flag’s as follows (in this
case we override the default 64MB buffer value with 32MB):

::

   {
       'mpi':
       {
           'enabled': True,
           'custom_mpi_options': '-x HOROVOD_FUSION_THRESHOLD=33554432'
       }
   }

You can also tweak batch cycles using ``HOROVOD_CYCLE_TIME`` parameter.
Note that cycle time is defined in miliseconds:

::

   {
       'mpi':
       {
           'enabled': True,
           'custom_mpi_options': '-x HOROVOD_CYCLE_TIME=5'
       }
   }

Optimizing MXNet Model
----------------------

Another optimization technique is related to optimizing the MXNet model
itself. It is recommended you first run the code with
``os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'`` Then you can copy
the best OS environment variables for future training. In our testing we
found the following to be the best results:

::

   os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
   os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
   os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
   os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
   os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
   os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

In Conclusion
-------------

In this post, we demonstrated how to reduce training time with Horovod
and Apache MXNet on Amazon SageMaker. Using Amazon SageMaker with
Horovod and MXNet, you can train your model out-of-the-box without
worrying about any additional complexities. For more information about
deep learning and MXNet, see the `MXNet crash
course <https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/getting-started/crash-course/index.html>`__
and `Dive into Deep Learning book <https://d2l.ai/>`__. You can also get
started on the `MXNet website <https://mxnet.apache.org/>`__ and MXNet
GitHub `examples
directory <https://github.com/apache/incubator-mxnet/tree/master/example>`__.
If you’re new to distributed training and want to dive deeper, we highly
recommend reading the paper `Horovod: fast and easy distributed deep
learning inTensorFlow <https://arxiv.org/pdf/1802.05799.pdf>`__. If you
are user of the Amazon Deep Learning Containers and AWS Deep Learning
AMIs, you can learn how to set up this workflow in that environment in
our recent blog post `how to run distributed training using Horovod and
MXNet on AWS DL containers and AWS Deep Learning
AMIs <https://aws.amazon.com/blogs/machine-learning/horovod-mxnet-distributed-training/>`__.

