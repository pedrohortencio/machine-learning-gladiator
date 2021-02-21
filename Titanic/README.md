# Titanic Dataset - A Comparison of Different Machine Learning Models

## About the data

The Titanic Dataset is a classic dataset and often explored as an introduction to Machine Learning. In this comparison, I'm using the data 
from [Kaggle's Titanic - Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic). It contains  two similar datasets (train, with labels, and test, without labels)
that include passenger information like name, age, gender, socio-economic class, etc.

The reason why this dataset is so popular is because while there was some element of luck involved in surviving,
it seems some groups of people were more likely to survive than others.

The challenge set by this dataset, as said by Kaggle, is to answer the question:  “what sorts of people were more likely to survive?”. 

With that in mind, my real question is: how good are different ML algorithms? How the accuracy is impacted with the use of normalized and unnormalized data?

## The algorithms tested

An important note is that I tried to give all models their best shot. That means that I tried different configurations for the Neural Network and used gridsearch for the sklearn and XGBoost models.
What that doesn't mean is that I didn't go as far as using state-of-the-art models and solutions. I didn't do an extensive EDA, too. Those little steps and data/feature engineering
could (and problably would) improve the results.

All models where trained and evaluated in the same subsets of the dataset.

The models are:

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
  
  
### The results

Overall, the best ML algorithm to solve this particular challenge was the XGBoost. That's not quite a surprise, as it is one of the most used models in Kaggle for nonperceptual data. 
Surprisingly, though, it had a slightly worse accuracy when normalized data was used in the training and testing. The decision tree and random forest, on the other hand, had
a slight increase in the accuracy. It was a behavior consistent in multiple runnings.

The neural network had difficulties in generalizing. Often, it overfitted fast. The main reason, in my opinion, is the small amount of data in this dataset, 
which causes more harm in neural networks' accuracy than in gradient boosting machines.

#### Visualizations

The accuracy scores when training and testing with nunnormalized data was the following:
![Comparison Between the Models using Unnormalized Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Titanic/accuracy-nonnormalized.png)
*Comparison Between the Models using Unnormalized Data*


The accuracy scores when using normalized data:
![Comparison Between the Models using Normalized Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Titanic/accuracy-normalized.png)


A direct comparison, for easier visualization:
![Comparison Between the Validation Accuracy using Normalized and Unnormalized Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/Titanic/comparison-normalized-nonnormalized.png)
*Comparison Between the Validation Accuracy using Normalized and Unnormalized Data*

The Bernoulli model had a significant drop in performance with normalized data, while the others had less dramatic oscillations.
