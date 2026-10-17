import pytest
import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def loaded_pipeline():
    return joblib.load("models/adult_income_pipeline.joblib")


@pytest.fixture(scope="session")
def X_val():
    adult = fetch_ucirepo(id=2)

    X = adult.data.features.copy()
    y = adult.data.targets.iloc[:, 0].copy()

    df = X.copy()
    df["income"] = y.to_numpy()

    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    df["income"] = (
        df["income"].str.strip().str.rstrip(".")
    )
    df = df.drop_duplicates().copy()

    X = df.drop(columns="income")
    y = df["income"]

    _, X_temp, _, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, _, _, _ = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Match the missing-value representation used by sklearn.
    X_val = X_val.replace({pd.NA: np.nan})

    return X_val
