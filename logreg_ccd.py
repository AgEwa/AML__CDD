import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def evaluate_model(y_true, y_pred, y_scores, metric="balanced_accuracy"):
    '''
    Evaluates predictions using the specified metric. Supported metrics are:
    ['recall', 'accuracy', 'precision', 'f1', 'balanced_accuracy', 'auc_roc', 'auc_pr']
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_scores (array-like): Predicted probabilities.
        metric (str): The metric to evaluate. Default is "balanced_accuracy".

    Returns:
        float: The computed metric value.
    '''
    if metric == 'recall':
        return recall_score(y_true, y_pred)
    elif metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred)
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'auc_roc':
        return roc_auc_score(y_true, y_scores)
    elif metric == 'auc_pr':
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class LogRegCCD:
    '''
    Logistic Regression using Coordinate Descent with L1 regularization.
    
    Attributes:
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        best_lambda (float): Best lambda value found during training.
        best_beta (array-like): Coefficients corresponding to the best lambda.
        train_mean (float): Mean of the training data.
        train_std (float): Standard deviation of the training data.
    '''


    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.best_lambda = None
        self.best_beta = None
        self.train_mean = 1
        self.train_std = 1
        

    def S(self, z, gamma):
        '''
        Soft-thresholding operator for L1 regularization.
        
        Args:
            z (float): Input value.
            gamma (float): Regularization parameter.
        
        Returns:
            float: Soft-thresholded value.
        '''
        return np.sign(z) * max(np.abs(z) - gamma, 0)

    def prepare_train_data(self, X_train):
        '''
        Standardize the training data and add a column of ones for the intercept.

        Args:
            X_train (array-like): Training features.

        Returns:
            array-like: Standardized training features with intercept.
        '''
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std = np.std(X_train, axis=0) + 1e-8
        X_train_stand = (X_train - self.train_mean) / self.train_std
        X_train_ready = np.hstack([np.ones((X_train_stand.shape[0], 1)), X_train_stand])
        return X_train_ready

    def prepare_test_data(self, X_test):
        '''
        Standardize the test data using the training mean and std, and add a column of ones for the intercept.
        
        Args:
            X_test (array-like): Test features.
            
        Returns:
            array-like: Standardized test features with intercept.
        '''
        X_test_stand = (X_test - self.train_mean) / self.train_std
        return np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    def coordinate_descent(self, X, y, lambd, beta):
        '''
        Performs coordinate descent to update the coefficients.
        
        Args:
            X (array-like): Feature matrix.
            y (array-like): Target variable.
            lambd (float): Regularization parameter.
            beta (array-like): Current coefficients.
        
        Returns:
            array-like: Updated coefficients.
        '''
        N, p = X.shape
        beta_new = beta.copy()
        for j in range(1, p):
            residual = y - X @ beta + beta[j] * X[:, j]
            rho = (1 / N) * np.sum(X[:, j] * residual)
            beta_new[j] = self.S(rho, lambd)
        return beta_new

    def fit(self, X_train, y_train, X_valid, y_valid, lambdas, metric="balanced_accuracy"):
        '''
        Fits the model using coordinate descent on different lambda values. It chooses the best lambda based on the specified metric.
        Supported metrics are:
        ['recall', 'accuracy', 'precision', 'f1', 'balanced_accuracy', 'auc_roc', 'auc_pr']

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_valid (array-like): Validation features.
            y_valid (array-like): Validation labels.
            lambdas (list): List of lambda values to evaluate.
            metric (str): The metric to evaluate. Default is "balanced_accuracy".
        '''
        X_train = self.prepare_train_data(X_train)
        X_valid = self.prepare_test_data(X_valid)
        N, p = X_train.shape
        best_metric_value = -np.inf
        beta_inter = np.mean(y_train)

        for lambd in lambdas:
            beta = np.ones(p)
            beta[0] = beta_inter
            for _ in range(self.max_iter):
                beta_old = beta.copy()
                p_tilde = expit(X_train @ beta)
                z = X_train @ beta + (y_train - p_tilde) / 0.25
                beta = self.coordinate_descent(X_train, z, lambd, beta)
                if np.linalg.norm(beta - beta_old, ord=2) < self.tol:
                    break

            probs = expit(X_valid @ beta)
            preds = (probs > 0.5).astype(int)
            metric_value = evaluate_model(y_valid, preds, probs, metric)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                self.best_beta = beta
                self.best_lambda = lambd


    def validate(self, X_valid, y_valid, metric="balanced_accuracy"):
        '''
        Calculates the specified metric for the validation set using the best beta found during training.
        
        Args:
            X_valid (array-like): Validation features.
            y_valid (array-like): Validation labels.
            metric (str): The metric to evaluate. Default is "balanced_accuracy".
        
        Returns:
            float: The computed metric value.
        '''
        if self.best_beta is None:
            raise ValueError("Model is not trained yet")

        X_valid = self.prepare_test_data(X_valid)
        probs = expit(X_valid @ self.best_beta)
        preds = (probs > 0.5).astype(int)
        return evaluate_model(y_valid, preds, probs, metric)

    def predict_proba(self, X_test):
        '''
        Predicts probabilities for the test set using the best beta found during training.
        
        Args:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted probabilities for the test set.
        '''
        if self.best_beta is None:
            raise ValueError("Model is not trained yet")
        return expit(X_test @ self.best_beta)

    def plot(self, X_train, y_train, X_valid, y_valid, lambdas, metric="balanced_accuracy"):
        '''
        Plots metric vs lambda values for the given lambdas and metric. Supported metrics are:
        ['recall', 'accuracy', 'precision', 'f1', 'balanced_accuracy', 'auc_roc', 'auc_pr']
        
        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_valid (array-like): Validation features.
            y_valid (array-like): Validation labels.
            lambdas (list): List of lambda values to evaluate.
            metric (str): The metric to evaluate. Default is "balanced_accuracy".
        '''
        scores = []
        for lambd in lambdas:
            self.fit(X_train, y_train, X_valid, y_valid, [lambd])
            score = self.validate(X_valid, y_valid, metric)
            scores.append(score)
        plt.plot(lambdas, scores, marker='o')
        plt.xlabel("Lambda")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Lambda")
        plt.show()

    def plot_coefficients(self, lambdas, X_train, y_train, metric="balanced_accuracy"):
        '''
        Plots values of coefficients for different lambda values.
        
        Args:
            lambdas (list): List of lambda values to evaluate.
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            metric (str): The metric to evaluate. Default is "balanced_accuracy".
        '''
        coefficients = []
        for lambd in lambdas:
            self.fit(X_train, y_train, X_train, y_train, [lambd], metric=metric)
            coefficients.append(self.best_beta.copy()[1:])
        coefficients = np.array(coefficients)
        for i in range(coefficients.shape[1]):
            plt.plot(lambdas, coefficients[:, i], label=f'Feature {i}')
        plt.xlabel("Lambda")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficients based on lambdas")
        plt.legend()
        plt.show()
