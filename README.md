# Machine Learning Gladiator - Battle Between Algorithms
Repository containing comparisons between different out-of-the-box machine learning algorithms in some classic datasets.

# Table of Contents

* [The Concept](https://github.com/pedrohortencio/machine-learning-gladiator#the-concept)
* [The Data](https://github.com/pedrohortencio/machine-learning-gladiator#the-data)
  * [MNIST](https://github.com/pedrohortencio/machine-learning-gladiator#mnist)
  * [Fashion MNIST](https://github.com/pedrohortencio/machine-learning-gladiator#fashion-mnist)
  * [Titanic](https://github.com/pedrohortencio/machine-learning-gladiator#titanic)
  * [Wine Quality](https://github.com/pedrohortencio/machine-learning-gladiator#wine-quality)
* [Roadmap](https://github.com/pedrohortencio/machine-learning-gladiator#roadmap)


## The Concept

The name is a sort of joke for a very established  concept: to try different classic algorithms in a dataset to experiment and see how each one behaves.

It's an attempt to answer some questions:

 * How does different models compare to each other in equal conditions?
 * What's the impact of fine-tuning (using Grid Search or other methods) the models?
 * What's the impact of data normalization in different models?
 * Is it ok to use out-of-the-box scikit-learn's models?
 * When to use Machine Learning after all? Are there situations where is better not to use ML models?

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

#### MNIST Models' Scores
One surprise is how well KNN model did in comparison with other more robust methods.
![Models' accuracy on test data (MNIST)](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/MNIST/Accuracy-test.png)
*Accuracy on test data with out-of-the-box sklearn algorithms. No optimization was made. The training data was a subset of 5000 random images from the MNIST dataset, and the test data was a subset of 2000 random images.*

The notebook is avaiable in the [MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/blob/main/MNIST/MNIST_Gladiator.ipynb) and also on [Google Colab](https://colab.research.google.com/github/pedrohortencio/machine-learning-gladiator/blob/main/Fashion%20MNIST/Fashion_MNIST_Gladiator.ipynb).

------------

### Fashion MNIST

The models used to predict the class of the Fashion MNIST's clothes are:

 * Deep Neural Network
 * Convolutional Neural Network
 * CNN with Transfer Learning from the VGG16 Model

#### Fashion MNIST Models' Scores

No surprises here. The CNN model did better than the DNN and the Transfer Learning model did (marginally) better than the CNN one.
![Comparison Between Models in the Fashion MNIST dataset](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Fashion%20MNIST/accuracy-comparison.png)
*All models were trained, validated and tested with the same datasets.*

The notebook is avaiable in the [Fashion MNIST subfolder](https://github.com/pedrohortencio/machine-learning-gladiator/tree/main/Fashion%20MNIST) and also on [Google Colab](https://colab.research.google.com/github/pedrohortencio/machine-learning-gladiator/blob/main/Fashion%20MNIST/Fashion_MNIST_Gladiator.ipynb).

------------

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


#### Titanic Models' Scores
Here is the final plot, a comparison of the validation accuracy score both with normalized and unnormalized data:

![Comparison Between the Validation Accuracy using Normalized and Unnormalized Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Titanic/comparison-normalized-nonnormalized.png)
*Comparison Between the Validation Accuracy using Normalized and Unnormalized Data*

------------

### Wine Quality
The Wine Quality Dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. The data can be found in [Kaggle (only the Red Wine variant)](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

The task can either be classification or regression. For this comparison I chose to treat the dataset as a classification problem, to see if the wine is good (quality equal or above 7) or bad (otherwise).

The dataset is not balanced, because there are many more examples of normal wines than of excellent and poor ones, and the authors of the dataset state that "we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.", so a robust EDA could improve the results (or straight up replace the ML models).

#### Wine Quality Models' Scores

This is a good examples of a scenario where the use of Machine Learning may not be the best path to take. Using out-of-the-box algorithms and only performing normalization in the data (without an extensive exploratory data analysis/feature engineering) yelded the results bellow. The high scores of the dummy classifier indicates that the logic behind the classification could be extracted without training and deploying a sofisticated machine learning model. 

![Wine Quality Models' Accuracy Comparison in Validation and Training (Cross Validation)](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Wine%20Quality/comparison-normalized-unnormalized.png)
*Wine Quality Models' Accuracy Comparison in Validation and Training (Cross Validation)*


## Roadmap

I plan to do similar analysis with House Prices and TMDB Box Office datasets.
