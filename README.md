# Machine Learning Gladiator - Battle Between Algorithms
Repository containing comparisons between different out-of-the-box machine learning algorithms in some classic datasets.

## The Concept

The name is a sort of joke for a very established  concept: to try different classic algorithms in a dataset to experiment and see how each one behaves.

It's an attempt to answer some questions:

 * How does different models compare to each other in equal conditions?
 * What's the impact of fine-tuning (using Grid Search or other methods) the models?
 * What's the impact of data normalization in different models?
 * Is it ok to use out-of-the-box scikit-learn's models?

The idea for this project came from Elite Data Science's blog, in this particular [list of projects](https://elitedatascience.com/machine-learning-projects-for-beginners). The concept of a gladiator, with several algorithms competing on similar terms, is fairly commomn, so there's other sources that use this term.

## The Data

Bellow is a brief of the datasets, the models and the final scores. More information, further conclusions and analysis can be found in the READMEs inside their subfolders in this repository.

### MNIST

The first dataset that I'll apply this concept is the well known MNIST dataset. As the classes are balanced, the metric used to evaluate the models is accuracy. The models created were:

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

#### MNIST Models's Scores
One surprise is how well KNN model did in comparison with other more robust methods.
![Models' accuracy on test data (MNIST)](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/MNIST/Accuracy-test.png)
*Accuracy on test data with out-of-the-box sklearn algorithms. No optimization was made. The training data was a subset of 5000 random images from the MNIST dataset, and the test data was a subset of 2000 random images.*

The notebook is avaiable in the [MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/blob/main/MNIST/MNIST_Gladiator.ipynb) and also on [Google Colab](https://colab.research.google.com/github/pedrohortencio/machine-learning-gladiator/blob/main/Fashion%20MNIST/Fashion_MNIST_Gladiator.ipynb).

### Fashion MNIST

The models used to predict the class of the Fashion MNIST's clothes are:

 * Deep Neural Network
 * Convolutional Neural Network
 * CNN with Transfer Learning from the VGG16 Model

#### Fashion MNIST Models's Scores

No surprises here. The CNN model did better than the DNN and the Transfer Learning model did (marginally) better than the CNN one.
![Comparison Between Models in the Fashion MNIST dataset](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Fashion%20MNIST/accuracy-comparison.png)
*All models were trained, validated and tested with the same datasets.*

The notebook is avaiable in the [Fashion MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/tree/main/Fashion%20MNIST) and also on [Google Colab](https://colab.research.google.com/github/pedrohortencio/machine-learning-gladiator/blob/main/Fashion%20MNIST/Fashion_MNIST_Gladiator.ipynb).

### Titanic
For this classic dataset, the models created were:

  * Dummy Classifier - an instance of sklearn that is used purely to stablish a baseline to the real models.
  * Bernoulli Naive Bayes
  * K-Nearest Neighbors Classifier
  * SGD Classifier
  * Logistic Regression
  * Ridge Classifier
  * Support Vector Classifier
  * Decision Tree
  * Random Forest
  * XGBoost Classifier
  * Neural Network (Deep Neural Network)


#### Titanic Models's Scores
Here is the final plot, a comparison of the validation accuracy score both with normalized and unnormalized data:

![Comparison Between the Validation Accuracy using Normalized and Unnormalized Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Titanic/comparison-normalized-nonnormalized.png)
*Comparison Between the Validation Accuracy using Normalized and Unnormalized Data*


## Roadmap

I plan to do similar analysis with Wine Quality and TMDB Box Office datasets.
