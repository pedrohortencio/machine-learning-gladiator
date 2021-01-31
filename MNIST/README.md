# Using the MNIST Dataset to Evaluate Scikit-Learn Models

## The Metodology

For this project, I used the Kaggle's MNIST dataset to train and evaluate several out-of-the-box scikit-learn models.

The models used are the following:

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

All models were trained using the same dataset: a subset of 5000 training images from Kaggle. The images were scaled to 0-1 intervals.

All models were used with as little modifications as possible. Whenever was possible, no parameters was changed. Some exceptions occured:
- SVC model required a change to the ```decision_function_shape``` parameter to allow one-vs-one decisions. The ```cache_size``` parameter was also changed to decrease training time.
- Logistic Regression model required a change to the ```max_iter``` parameter to allow convergence.

## The results

### Out-of-the-box Algorithms

The results are the following:

![Out-Of-The-Box Models' Accuracy on Test Data](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/MNIST/Accuracy-test.png)
*Accuracy on test data with out-of-the-box sklearn algorithms. No optimization was made.*

Without surprise, the Support Vector Classifier is the best choice. SVMs are well know to be a robust type of machine learning model.

I would say that a surprise is the presence of the KNN model in the top 3, with a solid 93.2% accuracy. That's a good time to remember that the training dataset is only 5000 images and there's no optimization.

Speaking of optimization, I did Grid Search with the 4 best models and Logistic Regression, to see how much the accuracy would change. The results are in the plot bellow:

![Accuracy after Grid Search](https://raw.githubusercontent.com/pedrohortencio/machine-learning-gladiator/main/MNIST/Accuracy-test-ft.png)

Not much of a difference, but a increase in all cases. It's important to note that I didn't do a perfect job grid searching the algorithms, because of the ammount of time that that would require.

## The Bottom Line

This experiment proves, one more time, that there's no free lunch: there's not a single algorithm that would outperform other in all tasks. In a lot of cases, it would make sense to
not use a top notch, state-of-the-art algorithm to classify digits. One (or maybe two) simple algorithms, that require little to no modification, and are fast to train, could be good
for the job.

I did a much more refined version of the SVC algorithm for another project, with tons of optimizations and a much more robust set of parameters. I manage to achieve 98.12% accuracy on
Kaggle. I also did a Convolutional Neural Network with this same dataset, and achieved a accuracy of 99.11%. Those models required a lot of time and try-and-test. They can be found
in [my repository dedicated to the Digit Recognizer Competition](https://github.com/pedrohortencio/digit-recognizer-kaggle).
