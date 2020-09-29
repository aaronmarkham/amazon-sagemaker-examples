Using SageMaker debugger to monitor attentions in BERT model training
---------------------------------------------------------------------

`BERT <https://arxiv.org/abs/1810.04805>`__ is a deep bidirectional
transformer model that achieves state-of the art results in NLP tasks
like question answering, text classification and others. In this
notebook we will use `GluonNLP <https://gluon-nlp.mxnet.io/>`__ to
finetune a pretrained BERT model on the `Stanford Question and Answering
dataset <https://web.stanford.edu/class/cs224n/reports/default/15848195.pdf>`__
and we will use `SageMaker
Debugger <https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html>`__
to monitor model training in real-time.

The paper `Visualizing Attention in Transformer-Based Language
Representation Models [1] <https://arxiv.org/pdf/1904.02679.pdf>`__
shows that plotting attentions and individual neurons in the query and
key vectors can help to identify causes of incorrect model predictions.
With SageMaker Debugger we can easily retrieve those tensors and plot
them in real-time as training progresses which may help to understand
what the model is learning.

The animation below shows the attention scores of the first 20 input
tokens for the first 10 iterations in the training.

 Fig. 1: Attention scores of the first head in the 7th layer

[1] *Visualizing Attention in Transformer-Based Language Representation
Models*: Jesse Vig, 2019, 1904.02679, arXiv

.. code:: ipython3

    ! pip install smdebug

.. code:: ipython3

    import boto3
    import sagemaker
    
    boto_session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

SageMaker training
~~~~~~~~~~~~~~~~~~

The following code defines the SageMaker Estimator. The entry point
script `train.py <entry_point/train.py>`__ defines the model training.
It downloads a BERT model from the GluonNLP model zoo and finetunes the
model on the Stanford Question Answering dataset. The training script
follows the official GluonNLP
`example <https://github.com/dmlc/gluon-nlp/blob/v0.8.x/scripts/bert/finetune_squad.py>`__
on finetuning BERT.

For demonstration purposes we will train only on a subset of the data
(``train_dataset_size``) and perform evaluation on a single batch
(``val_dataset_size``).

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
    
    role = sagemaker.get_execution_role()
    
    BUCKET_NAME = sagemaker_session.default_bucket()
    LOCATION_IN_BUCKET = 'smdebug-output'
    s3_bucket_for_tensors = 's3://{BUCKET_NAME}/{LOCATION_IN_BUCKET}'.format(BUCKET_NAME=BUCKET_NAME, LOCATION_IN_BUCKET=LOCATION_IN_BUCKET)
    
    mxnet_estimator = MXNet(entry_point='train.py',
                                source_dir='entry_point',
                                role=role,
                                train_instance_type='ml.p3.2xlarge',
                                train_instance_count=1,
                                framework_version='1.6.0',
                                py_version='py3',
                                hyperparameters = {'epochs': 3, 
                                                   'batch_size': 16,
                                                   'learning_rate': 5e-5,
                                                   'train_dataset_size': 1024,
                                                   'val_dataset_size': 16},
                                debugger_hook_config = DebuggerHookConfig(
                                  s3_output_path=s3_bucket_for_tensors,  
                                  collection_configs=[
                                    CollectionConfig(
                                        name="all",
                                        parameters={"include_regex": 
                                                    ".*multiheadattentioncell0_output_1|.*key_output|.*query_output",
                                                    "train.save_steps": "0",
                                                    "eval.save_interval": "1"}
                                        )
                                     ]
                                   )
                                )                                            

SageMaker Debugger provides default collections for gradients, weights
and biases. The default ``save_interval`` is 100 steps. A step presents
the work done by the training job for one batch (i.e. forward and
backward pass).

In this example we are also interested in attention scores, query and
key output tensors. We can emit them by just defining a new
`collection <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#collection>`__.
In this example we call the collection ``all`` and define the
corresponding regex. We save every iteration during validation phase
(``eval.save_interval``) and only the first iteration during training
phase (``train.save_steps``).

We also add the following lines in the validation loop to record the
string representation of input tokens:

.. code:: python

   if hook.get_collections()['all'].save_config.should_save_step(modes.EVAL, hook.mode_steps[modes.EVAL]):  
      hook._write_raw_tensor_simple("input_tokens", input_tokens)

.. code:: ipython3

    mxnet_estimator.fit(wait=False)

We can check the S3 location of tensors:

.. code:: ipython3

    path = mxnet_estimator.latest_job_debugger_artifacts_path()
    print('Tensors are stored in: {}'.format(path))

Get the training job name:

.. code:: ipython3

    job_name = mxnet_estimator.latest_training_job.name
    print('Training job name: {}'.format(job_name))
    
    client = mxnet_estimator.sagemaker_session.sagemaker_client
    
    description = client.describe_training_job(TrainingJobName=job_name)

We can access the tensors from S3 once the training job is in status
Training or Completed. In the following code cell we check the job
status.

.. code:: ipython3

    import time
    
    if description['TrainingJobStatus'] != 'Completed':
        while description['SecondaryStatus'] not in {'Training', 'Completed'}:
            description = client.describe_training_job(TrainingJobName=job_name)
            primary_status = description['TrainingJobStatus']
            secondary_status = description['SecondaryStatus']
            print('Current job status: [PrimaryStatus: {}, SecondaryStatus: {}]'.format(primary_status, secondary_status))
            time.sleep(15)

Get tensors and visualize BERT model training in real-time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we will retrieve the tensors of our training job and
create the attention-head view and neuron view as described in
`Visualizing Attention in Transformer-Based Language Representation
Models [1] <https://arxiv.org/pdf/1904.02679.pdf>`__.

First we create the
`trial <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#Trial>`__
that points to the tensors in S3:

.. code:: ipython3

    from smdebug.trials import create_trial
    
    trial = create_trial( path )

Next we import a script that implements the visualization for
attentation head view in Bokeh.

.. code:: ipython3

    from utils import attention_head_view, neuron_view
    from ipywidgets import interactive

We will use the tensors from the validation phase. In the next cell we
check if such tensors are already available or not.

.. code:: ipython3

    import numpy as np
    from smdebug import modes
    
    while (True):
        if len(trial.steps(modes.EVAL)) == 0:
            print("Tensors from validation phase not available yet")
        else:
            step = trial.steps(modes.EVAL)[0]
            break
        time.sleep(15) 

Once the validation phase started, we can retrieve the tensors from S3.
In particular we are interested in outputs of the attention cells which
gives the attention score. First we get the tensor names of the
attention scores:

.. code:: ipython3

    tensor_names = []
    
    for tname in sorted(trial.tensor_names(regex='.*multiheadattentioncell0_output_1')):
        tensor_names.append(tname)

Next we iterate over the available tensors of the validation phase. We
retrieve tensor values with
``trial.tensor(tname).value(step, modes.EVAL)``. Note: if training is
still in progress, not all steps will be available yet.

.. code:: ipython3

    steps = trial.steps(modes.EVAL)
    tensors = {}
    
    for step in steps:
        print("Reading tensors from step", step)
        for tname in tensor_names: 
            if tname not in tensors:
                tensors[tname]={}
            tensors[tname][step] = trial.tensor(tname).value(step, modes.EVAL)
    num_heads = tensors[tname][step].shape[1]

Next we get the query and key output tensor names:

.. code:: ipython3

    layers = []
    layer_names = {}
    
    for index, (key, query) in enumerate(zip(trial.tensor_names(regex='.*key_output_'), trial.tensor_names(regex='.*query_output_'))):
        layers.append([key,query])
        layer_names[key.split('_')[1]] = index

We also retrieve the string representation of the input tokens that were
input into our model during validation.

.. code:: ipython3

    input_tokens = trial.tensor('input_tokens').value(0, modes.EVAL)

Attention Head View
^^^^^^^^^^^^^^^^^^^

The attention-head view shows the attention scores between different
tokens. The thicker the line the higher the score. For demonstration
purposes, we will limit the visualization to the first 20 tokens. We can
select different attention heads and different layers. As training
progresses attention scores change and we can check that by selecting a
different step.

**Note:** The following cells run fine in Jupyter. If you are using
JupyterLab and encounter issues with the jupyter widgets (e.g. dropdown
menu not displaying), check the subsection in the end of the notebook.

.. code:: ipython3

    n_tokens = 20
    view = attention_head_view.AttentionHeadView(input_tokens, 
                                                 tensors,  
                                                 step=trial.steps(modes.EVAL)[0],
                                                 layer='bertencoder0_transformer0_multiheadattentioncell0_output_1',
                                                 n_tokens=n_tokens)

.. code:: ipython3

    interactive(view.select_layer, layer=tensor_names)

.. code:: ipython3

    interactive(view.select_head, head=np.arange(num_heads))

.. code:: ipython3

    interactive(view.select_step, step=trial.steps(modes.EVAL))

The following code cell updates the dictionary ``tensors`` with the
latest tensors from the training the job. Once the dict is updated we
can go to above code cell ``attention_head_view.AttentionHeadView`` and
re-execute this and subsequent cells in order to plot latest attentions.

.. code:: ipython3

    all_steps = trial.steps(modes.EVAL)
    new_steps = list(set(all_steps).symmetric_difference(set(steps)))
    
    for step in new_steps: 
        for tname in tensor_names:  
            if tname not in tensors:
                tensors[tname]={}
            tensors[tname][step] = trial.tensor(tname).value(step, modes.EVAL)

Neuron view
^^^^^^^^^^^

To create the neuron view as described in paper `Visualizing Attention
in Transformer-Based Language Representation Models
[1] <https://arxiv.org/pdf/1904.02679.pdf>`__, we need to retrieve the
queries and keys from the model. The tensors are reshaped and transposed
to have the shape: *batch size, number of attention heads, sequence
length, attention head size*

**Note:** The following cells run fine in Jupyter. If you are using
JupyterLab and encounter issues with the jupyter widgets (e.g. dropdown
menu not displaying), check the subsection in the end of the notebook.

.. code:: ipython3

    queries = {}
    steps = trial.steps(modes.EVAL)
    
    for step in steps:
        print("Reading tensors from step", step)
        
        for tname in trial.tensor_names(regex='.*query_output'):
           query = trial.tensor(tname).value(step, modes.EVAL)
           query = query.reshape((query.shape[0], query.shape[1], num_heads, -1))
           query = query.transpose(0,2,1,3)
           if tname not in queries:
                queries[tname] = {}
           queries[tname][step] = query

Retrieve the key vectors:

.. code:: ipython3

    keys = {}
    steps = trial.steps(modes.EVAL)
    
    for step in steps:
        print("Reading tensors from step", step)
        
        for tname in trial.tensor_names(regex='.*key_output'):
           key = trial.tensor(tname).value(step, modes.EVAL)
           key = key.reshape((key.shape[0], key.shape[1], num_heads, -1))
           key = key.transpose(0,2,1,3)
           if tname not in keys:
                keys[tname] = {}
           keys[tname][step] = key

We can now select different query vectors and see how they produce
different attention scores. We can also select different steps to see
how attention scores, query and key vectors change as training
progresses. The neuron view shows: \* Query \* Key \* Query x Key
(element wise product) \* Query \* Key (dot product)

.. code:: ipython3

    view = neuron_view.NeuronView(input_tokens, 
                                  keys=keys, 
                                  queries=queries, 
                                  layers=layers, 
                                  step=trial.steps(modes.EVAL)[0], 
                                  n_tokens=n_tokens,
                                  layer_names=layer_names)

.. code:: ipython3

    interactive(view.select_query, query=np.arange(n_tokens))

.. code:: ipython3

    interactive(view.select_layer, layer=layer_names.keys())

.. code:: ipython3

    interactive(view.select_step, step=trial.steps(modes.EVAL))

Note: Jupyter widgets in JupyterLab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter issues with this notebook in JupyterLab, you may have
to install JupyterLab extensions. You can do this by defining a
SageMaker `Lifecycle
configuration <https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html>`__.
A lifecycle configuration is a shell script that runs when you either
create a notebook instance or whenever you start an instance. You can
create a Lifecycle configuration directly in the SageMaker console (more
details
`here <https://aws.amazon.com/blogs/machine-learning/customize-your-amazon-sagemaker-notebook-instances-with-lifecycle-configurations-and-the-option-to-disable-internet-access/>`__)
When selecting ``Start notebook``, copy and paste the following code.
Once the configuration is created attach it to your notebook instance
and start the instance.

.. code:: sh

   #!/bin/bash

   set -e

   # OVERVIEW
   # This script installs a single jupyter notebook extension package in SageMaker Notebook Instance
   # For more details of the example extension, see https://github.com/jupyter-widgets/ipywidgets

   sudo -u ec2-user -i <<'EOF'

   # PARAMETERS
   PIP_PACKAGE_NAME=ipywidgets
   EXTENSION_NAME=widgetsnbextension

   source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv

   pip install $PIP_PACKAGE_NAME
   jupyter nbextension enable $EXTENSION_NAME --py --sys-prefix
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   # run the command in background to avoid timeout 
   nohup jupyter labextension install @bokeh/jupyter_bokeh &

   source /home/ec2-user/anaconda3/bin/deactivate

   EOF

