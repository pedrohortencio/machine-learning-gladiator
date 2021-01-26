# Machine Learning Gladiator - Battle Between Algorithms
Repository containing comparisons between different out-of-the-box machine learning algorithms in some classic datasets.

## The Concept

The name is nothing more than a gimmick for a very stablished concept: to try different classic algorithms in a dataset to experiment and see how each one behaves.
The idea to this project came from the Elite Data Science blog, in this particular [list of projects](https://elitedatascience.com/machine-learning-projects-for-beginners). The concept of a gladiator, with several algorithms competing on similar terms, is fairly commomn, so there's other sources that use this terminology.

## The Data

### MNIST

The first dataset that I'll apply this concept is the well known MNIST dataset. The metrics used to evaluate the performance of the models are: recall, precision, f1 score, accuracy and balanced accuracy. I'll primarily use scikit-learn to create and evaluate the models.
The notebook is avaiable in the [repository](https://github.com/pedrohortencio/machine-learning-gladiator/blob/main/MNIST/MNIST_Gladiator.ipynb) and also on [Google Colab](https://colab.research.google.com/drive/16XOZSnLCSpFp6HzXjo7OBrFLFrfEwHGY?usp=sharing).

## Roadmap

I plan to finish MNIST Gladiator with a bunch of classic algorithms: to K-Neighbors Classifier to Support Vector Classifier to Linear Regression. Some I know will do well, as SVC, while shome I have absolutely no idea how they'll behave (as Random Forest).

After that, I want to do similar things with both Fashion MNIST dataset, Titanic dataset and Wine Quality datasets.
