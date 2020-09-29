Streaming Algorithms in Machine Learning
========================================

In this notebook, we will use an extremely simple “machine learning”
task to learn about streaming algorithms. We will try to find the median
of some numbers in batch mode, random order streams, and arbitrary order
streams. The idea is to observe first hand the advantages of the
streaming model as well as to appreciate some of the complexities
involved in using it.

The task at hand will be to approximate the median (model) of a long
sequence of numbers (the data). This might seem to have little to do
with machine learning. We are used to thinking of a median, :math:`m`,
of number :math:`x_1,\ldots,x_n` in the context of statistics as the
number, :math:`m`, which is smaller than at most half the values
:math:`x_i` and larger than at most half the values :math:`x_i`.

Finding the median, however, also solves a proper machine learning
optimization problem (albeit a simple one). The median minimizes the
following clustering-like objective function

.. math:: m = \min_x \frac1n\sum_i|x - x_i|.

 In fact, the median is the solution to the well studied k-median
clustering problem in one dimension and :math:`k=1`. Moreover, the
extension to finding all quantiles is common in feature transformations
and an important ingedient in speeding up decission tree training.

Batch Algorithms
----------------

Let’s first import a few libraries and create some random data. Our data
will simple by :math:`100,000` equally spaced points between :math:`0`
and :math:`1`.

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    
    n = 100000
    data = np.linspace(0,1,n)
    np.random.shuffle(data)
    
    def f(x, data):
        return sum(abs(x - datum) for datum in data)/len(data)

Let’s look at the data to make sure everything is correct.

.. code:: ipython3

    %matplotlib inline
    # Plotting every 10th point
    plt.scatter(range(0,n,100),data[0:n:100],vmin=0,vmax=1.0)
    plt.ylim((0.0,1.0))
    plt.xlim((0,n))
    plt.show()

Computing the median brute force is trivial.

.. code:: ipython3

    from math import floor
    def batchMedian(data):
        n = len(data)
        median = sorted(data)[int(floor(n/2))]
        return(median)
    
    median = batchMedian(data)
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

The result is, of course, correct (:math:`0.5`). To get the median we
sorted the data in :math:`O(n\log n)` time even though QuickSelect would
have been faster (:math:`O(n)`). The algorithm speed is not the main
issue here though. The main drawback of this algorithm is that it must
store the entire dataset in memory. For either sorting or quickSelect
the algorithm must also duplicate the array. Binary search is also a
possible solution which doesn’t require data duplication but does
require :math:`O(\log(n))` passes over the data. When the data is large
this is either very expensive or simply impossible.

Streaming Algorithms (Random Order, SGD)
----------------------------------------

In the streaming model, we assume only an iterator over the data is
given. That is, we can only make one pass over the data. Moreover, the
algorithm is limited in its memory footprint and the limit is much lower
than the data size. Otherwise, we could “cheat” by storing all the data
in memory and executing the batch mode algorithm.

Gradient Descent (GD) type solutions are extremely common in this
setting and are, de facto, the only mechanism for optimizing neural
networks. In gradient descent, a step is taken in the direction opposite
of the gradient. In one dimension, this simply means going left if the
derivative is positive or right if the derivative is negative.

.. code:: ipython3

    %matplotlib inline
    xs = list(np.linspace(-1.0,2.0,50))
    ys = [f(x,data) for x in xs]
    plt.plot(xs,ys)
    plt.ylim((0.0,2.0))
    plt.xlim((-1.0,2.0))
    ax = plt.axes()
    ax.arrow(-0.5, 1.1, 0.3, -0.3, head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax.arrow(1.5, 1.1, -0.3, -0.3, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.show()

In **Stochastic Gradience Descent**, one only has a stochastic (random)
unbiased estimator of the gradient. So, instead of computing the
gradient of :math:`\frac1n\sum_i|x - x_i|` we can compute the gradient
of :math:`|x - x_i|` where :math:`x_i` is chosen **uniformly at random**
from the data. Note that a) the derivative of :math:`|x - x_i|` is
simply :math:`1` if :math:`x > x_i` and :math:`-1` otherwise and b) the
*expectation* of the derivative is exactly equal to the derivative of
the overall objective function.

Comment: the authors of the paper below suggest essentially this
algorithm but do not mention the connection to SGD for some reason.

Frugal Streaming for Estimating Quantiles: One (or two) memory suffices:
Qiang Ma, S. Muthukrishnan, Mark Sandler

.. code:: ipython3

    from math import sqrt
    
    def sgdMedian(data, learningRate=0.1, initMedianValue=0):
        median = initMedianValue
        for (t,x) in enumerate(data):
            gradient = 1.0 if x < median else -1.0
            median = median - learningRate*gradient/sqrt(t+1)
        return(median)
    
    median = sgdMedian(data, learningRate=0.1, initMedianValue=0)
    
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

The result isn’t exactly :math:`0.5` but it is pretty close. If this was
a real machine learning problem, matching the objective up to the 5th
digit of the true global minimum would have been very good.

Why does this work? Let’s plot our objective function to investigate
further.

It should not come as a big surprise to you that the objective function
is convex. After all, it is the sum of convex functions (absolute
values). It is a piece-wise linear curve that approximates a parabole in
the range :math:`(0,1)` and is linear outside that range. Therefore,
gradient descent is guaranteed to converge.

SGD significantly more efficient than sorting or even QuickSelect. More
importantly, its memory footprint is tiny, a handful of doubles,
*regardless of the size of the data*!!!

This is a huge advantage when operating with large datasets or with
limited hardware. Alas, SGD has some subtleties that make it a little
tricky to use sometimes.

.. code:: ipython3

    # SGD needs to be initialized carfully 
    median = sgdMedian(data, learningRate=0.1, initMedianValue=100.0)
    
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

.. code:: ipython3

    # SGD needs to set step sizes corectly (controled via the learing rate)
    median = sgdMedian(data, learningRate=0.001, initMedianValue=0.0)
    
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

These issues are usually alleviated by adaptive versions of SGD.
Enhancements to SGD such as second order (based) methods, adaptive
learning rate, and momentum methods may help in these situations but
still require tuning in many cases. A common approach is to use many
epochs.

.. code:: ipython3

    median=0.0
    numEpochs = 100
    for i in range(numEpochs):
        median = sgdMedian(data, learningRate=0.001, initMedianValue=median)
        
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

While clearly much less efficient than a single pass, increasing the
number of epochs seemed to have solved the problem. Machine learning
practitioners can relate to this result. That is, SGD is a great
algorithm IF one finds good parameters for initialization, learning
rate, number of epochs etc.

One of the main challenges in designing fundamentally better SGD-based
streaming algorithms is in adaptively controlling these parameters
during the run of the algorithm.

It is important to mention that there are also fundamentally better
algorithms than SGD for this problem. See for example:

*Sudipto Guha, Andrew McGregor* Stream Order and Order Statistics:
Quantile Estimation in Random-Order Streams. *SIAM J. Comput. 38(5):
2044-2059 (2009)*

Unfortunately, we don’t have time to dive into that…

Trending data poses a challenge…
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SGD and the other above algorithm have a fundamental drawback. They
inherently rely on the fact that the data is random. For SGD, the
gradient of the loss on a single point (or minibatch) must be an
estimator of the global gradient. This is not true if trends in data
make its statistics change (even slightly) over time. Let’s simulate
this with our data.

.. code:: ipython3

    %matplotlib inline
    
    # SGD also  depends on the data being reandomly suffeled
    n,k = len(data),10
    minibatches = [data[i:i+k] for i in range(0,n,k)]
    minibatches.sort(key=sum)
    trendyData = np.array(minibatches).reshape(n)
    
    # Plotting every 10th point in the trending dataset
    plt.scatter(range(0,n,100),trendyData[0:n:100],vmin=0,vmax=1.0)
    plt.ylim((0.0,1.0))
    plt.xlim((0,n))
    plt.show()

.. code:: ipython3

    median  = sgdMedian(trendyData, learningRate=0.1, initMedianValue=0.0)
    
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

Streaming Algorithms (single pass, arbitrary order)
---------------------------------------------------

One way not to be fooled by trends in data and to sample from it. The
algorithm uses Reservoir Sampling to obtain :math:`k` (in this case
:math:`k=1000`) uniformly chosen samples from the stream. Then, compute
the batch median of the sample.

The main drawback of sampling is that we now use more memory. Roughly
the sample size :math:`k` (:math:`k=1000` here). This much more than
:math:`O(1)` needed for SGD. Yet, it has some very appealing properties.
Sampling very efficient (:math:`O(1)` per update), it is very simple to
implement, it doesn’t have any numeric sensitivities or tunable input
parameters, and it is provably correct.

*(For the sake of simplicity below we use python’s builtin sample
function rather than recode reservoir sampling)*

.. code:: ipython3

    from random import sample 
    def sampleMedian(data):
        k=300
        samples = sample(list(data),k)
        return batchMedian(samples)
    
    median = sampleMedian(trendyData)
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

As you can see, sampling provides relatively good results.

Nevertheless, there is something deeply dissatisfying about it. The
algorithm was given :math:`100,000` points and used on :math:`1,000` of
them. I other words, it would have been just as accurate had we
collected only :math:`1\%` of the data.

Can we do better? Can an algorithm simultaneously take advantage of all
the data, have a fixed memory footprint, and not be sensitive to the
order in which the data is consumed? The answer is *yes!*. These are
known in the academic literature as Sketching (or simply streaming)
algorithms.

Specifically for approximating the median (or any other quantile), there
is a very recent result that shows how best to achieve that:

*Zohar S. Karnin, Kevin J. Lang, Edo Liberty* Optimal Quantile
Approximation in Streams. FOCS 2016: 71-78

The following code is a hacky version of the algorithm described in the
paper above. Warning: this function will not work for streams much
longer than :math:`100,000`!

.. code:: ipython3

    from kll300 import KLL300
    from bisect import bisect
    
    def sketchMedian(data):
        sketch = KLL300()
        for x in data:
            sketch.update(x)
            assert sketch.size <= 300 # making sure there is no cheating involved...
        items, cdf = sketch.cdf()
        i = bisect(cdf, 0.5)
        median = items[i]
        return median
    
    median = sketchMedian(trendyData)
    print('The median found if {}'.format(median))
    print('The objective value is {}'.format(f(median,data)))

Note that sketchMedian and sampleMedian both retain at most 300 items
from the stream. Still, the sketching solution is significantly more
accurate. Note that both sampling and sketching are randomized
algorithms. It could be that sampling happens to be more accurate than
sketching for any single run. But, as a whole, you should expect the
sketching algorithm to be much more accurate.

If you are curious about what sketchMedian actually does, you should
look here: \* Academic paper - https://arxiv.org/abs/1603.05346 \* JAVA
code as part of datasketches -
https://github.com/DataSketches/sketches-core/tree/master/src/main/java/com/yahoo/sketches/kll
\* Scala code by Zohar Karnin -
https://github.com/zkarnin/quantiles-scala-kll \* Python experiments by
Nikita Ivkin - https://github.com/nikitaivkin/quantilesExperiments

The point is, getting accurate and stable streaming algorithms is
complex. This is true even for very simple problems (like the one
above). But, if one can do that, the benefits are well worth it.
