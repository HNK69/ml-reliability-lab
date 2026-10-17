import numpy as np

def test_prediction_shape(loaded_pipeline, X_val):
    probs = loaded_pipeline.predict_proba(X_val)[:, 1]
    assert len(probs) == len(X_val)

def test_probabilities_are_valid(loaded_pipeline, X_val):
    probs = loaded_pipeline.predict_proba(X_val)[:, 1]
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0) & (probs <= 1))

def test_predictions_are_binary(loaded_pipeline, X_val, threshold=0.39):
    probs = loaded_pipeline.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    assert set(np.unique(preds)).issubset({0, 1})

def test_predictions_are_reproducible(loaded_pipeline, X_val):
    first = loaded_pipeline.predict_proba(X_val)[:, 1]
    second = loaded_pipeline.predict_proba(X_val)[:, 1]
    assert np.allclose(first, second)

def test_missing_values_are_handled(loaded_pipeline, X_val):
    sample = X_val.head(20).copy()
    sample.iloc[0, sample.columns.get_loc("age")] = np.nan
    probs = loaded_pipeline.predict_proba(sample)[:, 1]
    assert np.all(np.isfinite(probs))
