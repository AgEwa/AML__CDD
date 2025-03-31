# Project 1 - Advanced Machine Learning

This repository provides an implementation of the Cyclic Coordinate Descent (CCD) algorithm for parameter estimation in regularized logistic regression with an L1 penalty (Lasso).

The implementation of the algorithm can be found in the `logreg_ccd.py` file.

## Example usage
```python
from logreg_ccd import LogRegCCD, evaluate_model, prepare_data, get_lambda_sequence
from real_data_experiments import get_data_splitted

# Split the data into training, validation and testing sets
X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splitted(filepath='data.csv')
# Standarize the data and add columns of ones for the intercept
X_train, X_valid, X_test = prepare_data(X_train, X_valid, X_test)
# Generate lambdas space to explore
lambdas = get_lambda_sequence(X_train, y_train, n_lambda=100, eps=0.001)

# Initialize the model
ccd = LogRegCCD()
# Fit the model - choose the best lambda based on metric on validation set
scores, coeffs = ccd.fit(X_train, y_train, X_valid, y_valid, lambdas, metric="accuracy")

# Validate the results for the best lambda found
score_ccd = ccd.validate(X_test, y_test, metric="accuracy")
# or predict probabilities for the best lambda found
proba = ccd.predict_proba(X_test)

# Plot lambda values vs metric value plot
ccd.plot(lambdas, scores, metric="accuracy")
# Plot lambda values vs coefficient values plot
ccd.plot_coefficients(lambdas, coeffs, metric="accuracy")
```

## Features and methods

### Data Splitting
The `get_data_splitted` function divides the dataset into training, validation, and testing subsets. This ensures the model is trained on one part of the data, validated on another to tune hyperparameters, and tested on a completely separate set to evaluate generalization performance. We assume that the dataset has no header, and the target variable is located in the last column.

### Data Preparation
The `prepare_data` function standardizes the data and adds a column of ones for the intercept.

### Generating Lambda Sequence
The `get_lambda_sequence` function creates a sequence of regularization parameter (`lambda`) values to be tested during model fitting. The `n_lambda` parameter controls the number of values, and eps defines the range.

### LogRegCCD initialization parameters

`tol` (tolerance) is a parameter that sets the convergence threshold for the algorithm - the algorithm will stop if the change in parameter values between iterations is smaller than this value.

`max_iter` parameter defines the maximum number of iterations the algorithm will perform, even if convergence is not reached before that.

### Fitting the model

The `fit` method searches through the provided `lambdas` (regularization parameter) values. For each value, it fits the coefficients on the training set. It then evaluates the perfomance based on the metric specified in the `metric` parameter using the validation set. The best lambda is selected based on the highest performance.

### Predicting probabilities

The `predict_proba` method returns the probability of each observation belonging to class 1.

### Validating the model

The `validate` method allows you to calculate the chosen performance metric. The following metrics are supported:

- "recall",
- "accuracy",
- "precision",
- "f1",
- "balanced_accuracy",
- "auc_roc",
- "auc_pr".

### Plotting and Visualization

The `LogRegCCD` class provides methods for visualizing the model's perfomance and coefficient behavior as a function of the regularization parameter.

1. Plotting Metric vs Lambda (method `plot`) <br>
This method plots the specified performance metric against different values of the regularization parameter.
<br>

![example plot](./readme_charts/plot.png)

2. Plotting Coefficients vs Lambda (method `plot_coefficients`) <br>
This method visualizes the behavior of the model's coefficients as a function of different values of the regularization parameter.
<br>

![example coefficient plot](./readme_charts/coefficient.png)
