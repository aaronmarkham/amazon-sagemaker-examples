Example R Notebook
==================

A simple R example in the Notebook to test the installation of the R
kernel functioned properly.

.. code:: 

    library(ggplot2)
    library(randomForest)

Bring in the Iris dataset: - 150 observations, 50 each for 3 types of
iris flowers - 4 features measuring length and width for sepals and
petals

.. code:: 

    data(iris)
    head(iris)

.. code:: 

    qplot(iris$Sepal.Length, geom='histogram')

Let’s fit a simple random forest model.

.. code:: 

    rf <- randomForest(Species ~ ., data=iris, importance=TRUE, proximity=TRUE)

Let’s check the accuracy of the model.

.. code:: 

    table(iris$Species, rf$predicted)
