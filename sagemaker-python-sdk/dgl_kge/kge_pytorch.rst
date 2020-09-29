Training knowledge graph embedding by using the Deep Graph Library with PyTorch backend
---------------------------------------------------------------------------------------

The **Amazon SageMaker Python SDK** makes it easy to train Deep Graph
Library (DGL) models. In this example, you generate knowledge graph
embedding using the `DMLC DGL API <https://github.com/dmlc/dgl.git>`__
and FB15k dataset.

For more details about Knowledge Graph Embedding and this example, see
https://github.com/dmlc/dgl/tree/master/apps/kg

Setup
~~~~~

Define a few variables that are needed later in the example.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # Setup session
    sess = sagemaker.Session()
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # IAM execution role that gives Amazon SageMaker access to resources in your AWS account.
    # You can use the Amazon SageMaker Python SDK to get the role from the notebook environment. 
    role = get_execution_role()

The Amazon SageMaker estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Amazon SageMaker estimator allows you to run single machine in
Amazon SageMaker, using CPU or GPU-based instances.

When you create the estimator, pass in the file name of the training
script and the name of the IAM execution role. Also provide a few other
parameters. train_instance_count and train_instance_type determine the
number and type of Amazon SageMaker instances that are used for the
training job. The hyperparameters parameter is a dictionary of values
that is passed to your training script as parameters so that you can use
argparse to parse them.

Here, you can directly use the DL Container provided by Amazon SageMaker
for training DGL models by specifying the PyTorch framework version (>=
1.3.1) and the python version (only py3). You can also add a task_tag
with value ‘DGL’ to help tracking the task.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    ENTRY_POINT = 'train.py'
    CODE_PATH = './'
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    
    params = {}
    params['dataset'] = 'FB15k'
    params['model'] = 'DistMult'
    params['batch_size'] = 1024
    params['neg_sample_size'] = 256
    params['hidden_dim'] = 2000
    params['gamma'] = 500.0
    params['lr'] = 0.1
    params['max_step'] = 100000
    params['batch_size_eval'] = 16
    params['valid'] = True
    params['test'] = True
    params['neg_adversarial_sampling'] = True
    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    
    estimator = PyTorch(entry_point=ENTRY_POINT,
                        source_dir=CODE_PATH,
                        role=role, 
                        train_instance_count=1, 
                        train_instance_type='ml.p3.2xlarge',
                        framework_version="1.3.1",
                        py_version='py3',
                        debugger_hook_config=False,
                        tags=task_tags,
                        hyperparameters=params,
                        sagemaker_session=sess)

Running the Training Job
~~~~~~~~~~~~~~~~~~~~~~~~

After you construct the Estimator object, fit it by using Amazon
SageMaker. The dataset is automatically downloaded.

.. code:: ipython3

    estimator.fit()

Output
------

You can get the resulting embedding output from the Amazon SageMaker
console by searching for the training task and looking for the address
of ‘S3 model artifact’
