Learning Tic-Tac-Toe with Reinforcement Learning
================================================

**Train with SageMaker RL and evaluate interactively within the
notebook**

--------------

--------------

Outline
-------

1.  `Overview <#Overview>`__
2.  `Setup <#Setup>`__
3.  `Code <#Code>`__
4.  `Environment <#Environment>`__
5.  `Preset <#Preset>`__
6.  `Launcher <#Launcher>`__
7.  `Train <#Train>`__
8.  `Deploy <#Deploy>`__
9.  `Inference <#Inference>`__
10. `Play <#Play>`__
11. `Wrap Up <#Wrap-Up>`__

--------------

Overview
--------

Tic-tac-toe is one of the first games children learn to play and was one
of the `first computer games
ever <https://en.wikipedia.org/wiki/OXO>`__. Optimal play through
exhaustive search is relatively straightforward, however, approaching
with a reinforcement learning agent can be educational.

This notebook shows how to train a reinforcement learning agent with
SageMaker RL and then play locally and interactively within the
notebook. Unlike SageMaker local mode, this method does not require a
docker container to run locally, instead using an endpoint and
integration with a small Jupyter app (*Note, this app does not work in
JupyterLab*).

--------------

Setup
-----

Let’s start by defining our S3 bucket and and IAM role.

.. code:: ipython3

    import sagemaker
    
    bucket = sagemaker.Session().default_bucket()
    role = sagemaker.get_execution_role()

Let’s import the libraries we’ll use.

.. code:: ipython3

    import os
    import numpy as np
    import sagemaker
    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
    from tic_tac_toe_game import TicTacToeGame

--------------

Code
----

Our tic-tac-toe example requires 3 scripts in order to train our agent
using SageMaker RL. The scripts are placed in the ``./src`` directory
which is sent to the container when the SageMaker training job is
initiated.

Environment
~~~~~~~~~~~

For our tic-tac-toe use case we’ll create a custom Gym environment. This
means we’ll specify a Python class which inherits from ``gym.Env`` and
has two methods: ``reset()`` and ``step()``. These will provide the
agent its state, actions, and rewards for learning. In more detail:

The ``__init__()`` method is called at the beginning of the SageMaker
training job and: 1. Starts the 3x3 tic-tac-toe board as a NumPy array
of zeros 1. Prepares the state space as a flattened version of the board
(length 9) 1. Defines a discrete action space with 9 possible options
(one for each place on the board)

The ``reset()`` method is called at the beginning of each episode and:
1. Clears the 3x3 board (sets all values to 0) 1. Does some minor
record-keeping for tracking across tic-tac-toe games

The ``step()`` method is called for each iteration in an episode and: 1.
Adjusts the board based on the action chosen by the agent based on the
previous state 1. Generates rewards based on performance 1.
Automatically chooses the move for the agent’s opponent if needed

Note: \* The opponent has not been programmed for perfect play. If we
taught our agent against a perfect opponent, it would not generalize to
scenarios where the rules of perfect play were not followed. \* If our
agent selects an occupied space, it is given a minor penalty (-0.1) and
asked to choose again. Although the state doesn’t change across these
steps (meaning the agent’s network’s prediction should stay the same),
randomness in the agent should eventually result in different actions.
However, if the agent chooses an occupied space 10 times in a row, the
game is forfeit. Selecting an action only from available spaces would
require more substantial modification than was desired for this example.
\* Other rewards only occur when a game is completed (+1 for win, 0 for
draw, -1 for loss). \* The board is saved as a NumPy array where a value
of +1 represents our agent’s moves (``X``\ s) and a value of -1
represents the opponent’s moves (``O``\ s).

.. code:: ipython3

    !pygmentize ./src/tic_tac_toe.py

Preset
~~~~~~

The preset file specifies Coach parameters used by our reinforcement
learning agent. For this problem we’ll use a `Clipped PPO
algorithm <https://nervanasystems.github.io/coach/components/agents/policy_optimization/cppo.html>`__.
We have kept the preset file deliberately spartan, deferring to defaults
for most parameters, in order to focus on just the key components.
Performance of our agent could likely be improved with increased tuning.

.. code:: ipython3

    !pygmentize ./src/preset.py

Launcher
~~~~~~~~

The launcher is a script used by Amazon SageMaker to drive the training
job on the SageMaker RL container. We have kept it minimal, only
specifying the name of the preset file to be used for the training job.

.. code:: ipython3

    !pygmentize ./src/train-coach.py

--------------

Train
-----

Now, let’s kick off the training job in Amazon SageMaker. This call can
include hyperparameters that overwrite values in ``train-coach.py`` or
``preset.py``, but in our case, we’ve limited to defining: 1. The
location of our agent code ``./src`` and dependencies in ``common``. 1.
Which RL and DL framework to use (SageMaker also supports `Ray
RLlib <https://ray.readthedocs.io/en/latest/rllib.html>`__ and Coach
TensorFlow). 1. The IAM role granted permissions to our data in S3 and
ability to create SageMaker training jobs. 1. Training job hardware
specifications (in this case just 1 ml.m4.xlarge instance). 1. Output
path for our checkpoints and saved episodes. 1. A single hyperparameter
specifying that we would like our agent’s network to be output (in this
case as an ONNX model).

.. code:: ipython3

    estimator = RLEstimator(source_dir='src',
                            entry_point="train-coach.py",
                            dependencies=["common/sagemaker_rl"],
                            toolkit=RLToolkit.COACH,
                            toolkit_version='0.11.0',
                            framework=RLFramework.MXNET,
                            role=role,
                            train_instance_count=1,
                            train_instance_type='ml.m4.xlarge',
                            output_path='s3://{}/'.format(bucket),
                            base_job_name='DEMO-rl-tic-tac-toe',
                            hyperparameters={'save_model': 1})
    
    estimator.fit()

--------------

Deploy
------

Normally we would evaluate our agent by looking for reward convergence
or monitoring performance across epsisodes. Other SageMaker RL example
notebooks cover this in detail. We’ll skip that for the more tangible
approach of testing the trained agent by playing against it ourselves.
To do that, we’ll first deploy the agent to a realtime endpoint to get
predictions.

Inference
~~~~~~~~~

Our deployment code: 1. Unpacks the ONNX model output and prepares it
for inference in ``model_fn`` 1. Generates predictions from our network,
given state (a flattened tic-tac-toe board) in ``transform_fn``

.. code:: ipython3

    !pygmentize ./src/deploy-coach.py

Endpoint
~~~~~~~~

Now we’ll actually create a SageMaker endpoint to call for predictions.

*Note, this step could be replaced by importing the ONNX model into the
notebook environment.*

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1, 
                                 instance_type='ml.m4.xlarge', 
                                 entry_point='deploy-coach.py')

--------------

Play
----

Let’s play our agent. After running the cell below, just click on one
the boxes to make your move. To restart the game, simply execute the
cell again.

*This cell uses the ``TicTacToeGame`` class from ``tic_tac_toe_game.py``
script to build an extremely basic tic-tac-toe app within a Jupyter
notebook. The opponents moves are generated by invoking the
``predictor`` passed at initialization. Please refer to the code for
additional details.*

.. code:: ipython3

    t = TicTacToeGame(predictor)
    t.start()

--------------

Wrap Up
-------

In this notebook we trained a reinforcement learning agent to play a
simple game of tic-tac-toe, using a custom Gym environment. It could be
built upon to solve other problems or improved by:

-  Training for more episodes
-  Using a different reinforcement learning algorithm
-  Tuning hyperparameters for improved performance
-  Or how about a nice game of `global thermonuclear
   war <https://youtu.be/s93KC4AGKnY?t=41>`__?

Let’s finish by cleaning up our endpoint to prevent any persistent
costs.

.. code:: ipython3

    predictor.delete_endpoint()
