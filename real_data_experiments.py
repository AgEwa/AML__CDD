import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from logreg_ccd import LogRegCCD, evaluate_model, get_lambda_sequence, prepare_data


def get_data_splited(filepath:str):
    X = pd.read_csv(filepath, header=None)
    y = X.iloc[:, -1]
    X = X.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def run_scenerio(X_train, X_valid, X_test, y_train, y_valid, y_test, lambdas, log_reg, metric, max_iter=10):
    print("Training with metric={}".format(metric))
    ccd = LogRegCCD(max_iter=max_iter)
    scores, coeffs = ccd.fit(X_train, y_train, X_valid, y_valid, lambdas, metric=metric)

    score_log_reg = evaluate_model(y_test, log_reg.predict(X_test), log_reg.predict_proba(X_test)[:, 1], metric=metric)
    print("Score of sklearn Logistic Regression with no penalty on test data:", score_log_reg)

    score_ccd = ccd.validate(X_test, y_test, metric=metric)
    print("Score of CCD Logistic Regression on test data:", score_ccd)

    print('Best lambda: {}'.format(ccd.best_lambda))
    print('Best beta: {}'.format(ccd.best_beta))

    ccd.plot(lambdas, scores, metric=metric)
    ccd.plot_coefficients(lambdas, coeffs, metric=metric)

    return ccd.best_lambda, ccd.best_beta, score_ccd, score_log_reg

def run_experiment(data_source_path,max_iter=10):
    results = {}
    for metric in ['auc_roc', 'auc_pr', 'f1', 'balanced_accuracy']:
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splited(data_source_path)
        X_train, X_valid, X_test = prepare_data(X_train, X_valid, X_test)
        log_reg = LogisticRegression(penalty=None).fit(X_train, y_train)
        lambdas = get_lambda_sequence(X_train, y_train,n_lambda=10, eps=0.01)
        ccd_best_lambda, ccd_best_beta, score_ccd, score_log_reg = run_scenerio(X_train, X_valid, X_test, y_train, y_valid, y_test, lambdas, log_reg, metric, max_iter)
        log_reg_best_beta = log_reg.coef_[0]
        results[metric] = {
            'best_lambda': ccd_best_lambda,
            'best_beta': ccd_best_beta.tolist(),
            'score_ccd': score_ccd,
            'score_log_reg': score_log_reg,
            'best_beta_log_reg': log_reg_best_beta.tolist()
        }
    return results

def compare_betas(beta_ccd, beta_log_reg):
    assert len(beta_ccd) == len(beta_log_reg)
    beta_ccd=np.array(beta_ccd)
    beta_log_reg=np.array(beta_log_reg)
    plt.figure(figsize=(10,6))
    colors = ['crimson' if np.abs(b_log-b_ccd)< 0.01*min(b_log, b_ccd) else ('orange' if b_log*b_ccd<0 else 'lawngreen')
              for b_log, b_ccd in zip(beta_log_reg, beta_ccd)]

    plt.scatter(beta_log_reg, beta_ccd, c=colors, alpha=0.6)
    plt.ylabel("CCD Coefficients")
    plt.xlabel("LogReg Coefficients")
    custom_elements = [mpatches.Patch(color='crimson', label='(almost) equal values'),
                       mpatches.Patch(color='orange', label='Opposite sign values'),
                       mpatches.Patch(color='lawngreen', label='Same sign values')]

    # Add the custom legend
    plt.legend(handles=custom_elements, loc='upper left')

    plt.show()

    ccd_non_zero_count = beta_ccd[beta_ccd != 0].shape[0]
    print(f"CCD non-zero beta count: {ccd_non_zero_count}, out of {len(beta_ccd)}")
    beta_log_reg_sorted_idx = np.argsort(np.abs(beta_log_reg), stable=True)
    beta_ccd_sorted_idx = np.argsort(np.abs(beta_ccd), stable=True)
    print(f"How many of ccd betas correspond also to top {ccd_non_zero_count} log reg betas: {len(np.intersect1d(beta_log_reg_sorted_idx[-ccd_non_zero_count:],beta_ccd_sorted_idx[-ccd_non_zero_count:]))}")
    return np.linalg.norm(beta_ccd - beta_log_reg)/len(beta_ccd)


