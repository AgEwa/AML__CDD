import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from logreg_ccd import LogRegCCD, evaluate_model, get_lambda_sequence, prepare_data


def get_data_splited(filepath:str):
    X = pd.read_csv(filepath, header=None)
    y = X.iloc[:, -1]
    X = X.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def run_scenerio(X_train, X_valid, X_test, y_train, y_valid, y_test, lambdas, log_reg, metric):
    print("Training with metric={}".format(metric))
    ccd = LogRegCCD(max_iter=1000)
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

def run_experiment(data_source_path):
    results = {}
    for metric in ['auc_roc', 'auc_pr', 'f1', 'balanced_accuracy']:
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_data_splited(data_source_path)
        X_train, X_valid, X_test = prepare_data(X_train, X_valid, X_test)
        log_reg = LogisticRegression(penalty=None).fit(X_train, y_train)
        lambdas = get_lambda_sequence(X_train, y_train,n_lambda=100, eps=0.001)
        ccd_best_lambda, ccd_best_beta, score_ccd, score_log_reg = run_scenerio(X_train, X_valid, X_test, y_train, y_valid, y_test, lambdas, log_reg, metric)
        results[metric] = {
            'best_lambda': ccd_best_lambda,
            'best_beta': ccd_best_beta,
            'score_ccd': score_ccd,
            'score_log_reg': score_log_reg
        }
    return results