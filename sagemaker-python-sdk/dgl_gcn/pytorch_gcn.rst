Training Amazon SageMaker models by using the Deep Graph Library with PyTorch backend
-------------------------------------------------------------------------------------

The **Amazon SageMaker Python SDK** makes it easy to train Deep Graph
Library (DGL) models. In this example, you train a simple graph neural
network using the `DMLC DGL API <https://github.com/dmlc/dgl.git>`__ and
the `Cora dataset <https://relational.fit.cvut.cz/dataset/CORA>`__. The
Cora dataset describes a citation network. The Cora dataset consists of
2,708 scientific publications classified into one of seven classes. The
citation network consists of 5,429 links. The task is to train a node
classification model using Cora dataset.

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
    # Feel free to specify a different bucket here.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # IAM execution role that gives Amazon SageMaker access to resources in your AWS account.
    # You can use the Amazon SageMaker Python SDK to get the role from the notebook environment. 
    role = get_execution_role()

The training script
~~~~~~~~~~~~~~~~~~~

The pytorch_gcn.py script provides all the code you need for training an
Amazon SageMaker model.

.. code:: ipython3

    !cat pytorch_gcn.py

SageMaker’s estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Amazon SageMaker Estimator allows you to run single machine in
Amazon SageMaker, using CPU or GPU-based instances.

When you create the estimator, pass in the filename of the training
script and the name of the IAM execution role. You can also provide a
few other parameters. train_instance_count and train_instance_type
determine the number and type of Amazon SageMaker instances that are
used for the training job. The hyperparameters parameter is a dictionary
of values that is passed to your training script as parameters so that
you can use argparse to parse them. You can see how to access these
values in the pytorch_gcn.py script above.

Here, you can directly use the DL Container provided by Amazon SageMaker
for training DGL models by specifying the PyTorch framework version (>=
1.3.1) and the python version (only py3). You can also add a task_tag
with value ‘DGL’ to help tracking the task.

For this example, choose one ml.p3.2xlarge instance. You can also use a
CPU instance such as ml.c4.2xlarge for the CPU image.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    CODE_PATH = 'pytorch_gcn.py'
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    
    params = {}
    params['dataset'] = 'cora'
    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    estimator = PyTorch(entry_point=CODE_PATH,
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.p3.2xlarge', # 'ml.c4.2xlarge '
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

You can get the model training output from the Amazon Sagemaker console
by searching for the training task named pytorch-gcn and looking for the
address of ‘S3 model artifact’
