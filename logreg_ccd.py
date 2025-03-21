import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def evaluate_model(y_true, y_pred, y_scores, metric="balanced_accuracy"):
    if metric == 'recall':
        return recall_score(y_true, y_pred)
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
    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.best_lambda = None
        self.best_beta = None

    def S(self, z, gamma):
        return np.sign(z) * max(np.abs(z) - gamma, 0)

    def prepare_train_data(self, X_train):
        '''
        Standardize the data and add a column of ones for the intercept.
        '''
        X_train_stand = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_train_ready = np.hstack([np.ones((X_train_stand.shape[0], 1)), X_train_stand])
        return X_train_ready

    def prepare_test_data(self, X_test):
        return np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    def coordinate_descent(self, X, y, lambd, beta):
        N, p = X.shape
        beta_new = beta.copy()
        for j in range(1, p):
            residual = y - X @ beta + beta[j] * X[:, j]
            rho = (1 / N) * np.sum(X[:, j] * residual)
            beta_new[j] = self.S(rho, lambd)
        return beta_new

    def fit(self, X_train, y_train, X_valid, y_valid, lambdas, metric="balanced_accuracy"):
        # I added the option to fit lambda on validation set (it's in the description of the project),
        # but in the example experiment I used the same set for training and validation
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
            preds = (X_valid @ beta > 0).astype(int)
            metric_value = evaluate_model(y_valid, preds, probs, metric)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                self.best_beta = beta
                self.best_lambda = lambd

    def validate(self, X_valid, y_valid, metric="balanced_accuracy"):
        if self.best_beta is None:
            raise ValueError("Model is not trained yet")

        X_valid = self.prepare_test_data(X_valid)
        probs = expit(X_valid @ self.best_beta)
        preds = (probs > 0.5).astype(int)
        return evaluate_model(y_valid, preds, probs, metric)

    def predict_proba(self, X_test):
        if self.best_beta is None:
            raise ValueError("Model is not trained yet")
        return 1 / (1 + np.exp(-X_test @ self.best_beta))

    def plot(self, X_train, y_train, X_valid, y_valid, lambdas, metric="balanced_accuracy"):
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
