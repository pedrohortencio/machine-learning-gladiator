# Machine Learning Gladiator - Battle Between Algorithms
Repository containing comparisons between different out-of-the-box machine learning algorithms in some classic datasets.

## The Concept

The name is nothing more than a gimmick for a very stablished concept: to try different classic algorithms in a dataset to experiment and see how each one behaves.
The idea to this project came from the Elite Data Science blog, in this particular [list of projects](https://elitedatascience.com/machine-learning-projects-for-beginners). The concept of a gladiator, with several algorithms competing on similar terms, is fairly commomn, so there's other sources that use this terminology.

## The Data

### MNIST

The first dataset that I'll apply this concept is the well known MNIST dataset. As the classes are balanced, the metric used to evaluate the models is accuracy. The models created are:

  * Support Vector Classifier
  * Random Forest
  * K-Nearest Neighbors Classifier
  * MLP Classifier
  * Gradient Boosting Classifier
  * Logistic Regression
  * Perceptron
  * Ridge Classifier
  * Ridge Classifier CV
  * Bernoulli Naive Bayes
  * Gaussian Naive Bayes
  * Decision Tree

#### MNIST Results
![Models' accuracy on test data (MNIST)](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/MNIST/Accuracy-test.png)
*Accuracy on test data with out-of-the-box sklearn algorithms. No optimization was made. The training data was a subset of 5000 random images from the MNIST dataset, and the test data was a subset of 2000 random images.*

The notebook is avaiable in the [MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/blob/main/MNIST/MNIST_Gladiator.ipynb) and also on [Google Colab](https://colab.research.google.com/drive/16XOZSnLCSpFp6HzXjo7OBrFLFrfEwHGY?usp=sharing).

### Fashion MNIST

The models used to predict the class of the Fashion MNIST's clothes are:

 * Deep Neural Network
 * Convolutional Neural Network
 * CNN with Transfer Learning from the VGG16 Model

#### Fashion MNIST Results

![Comparison Between Models in the Fashion MNIST dataset](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Fashion%20MNIST/accuracy-comparison.png)
*All models were trained, validated and tested with the same datasets.*

The notebook is avaiable in the [Fashion MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/tree/main/Fashion%20MNIST) and also on [Google Colab](https://colab.research.google.com/github/pedrohortencio/machine-learning-gladiator/blob/main/Fashion%20MNIST/Fashion_MNIST_Gladiator.ipynb).

## Roadmap

I want to do similar analysis with Titanic, Wine Quality and TMDB Box Office datasets.

The next one will probably be the Titanic. It'll be interesting to see how the results from scikit-learn Random Forest compares to XGBoost.
