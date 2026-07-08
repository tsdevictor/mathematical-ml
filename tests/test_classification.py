import numpy as np

from mathematical_ml.classification import HardMarginSVM


def test_hard_margin_svm_separates_simple_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([-1, -1, 1, 1])

    model = HardMarginSVM()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.mean(preds == y) == 1.0
    assert len(model.support_indices_) > 0
