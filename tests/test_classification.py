import numpy as np

from mathematical_ml.classification import LinearSVM, LogisticRegressionGD


def test_logistic_regression_fits_simple_data():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionGD(learning_rate=0.5, num_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.mean(preds == y) >= 0.75


def test_linear_svm_fits_simple_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([-1, -1, 1, 1])

    model = LinearSVM(learning_rate=0.1, num_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.mean(preds == y) == 1.0
