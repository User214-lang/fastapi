import pandas as pd
import pytest
from dataset import prepare_data, split_data, get_class_distribution

def test_prepare_data_returns_X_y_and_features(sample_df):
    X, y, numeric, categorical = prepare_data(sample_df)
    assert "churn" not in X.columns
    assert len(y) == len(sample_df)
    assert len(X) == len(sample_df)
    assert isinstance(numeric, list)
    assert isinstance(categorical, list)

def test_prepare_data_missing_churn_raises(sample_df):
    with pytest.raises(ValueError):
        prepare_data(sample_df.drop(columns=["churn"]))

def test_split_data_sizes(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    assert len(X_test) == 5
    assert len(X_train) == 15
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_split_data_stratify(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    _, _, y_train, y_test = split_data(X, y, test_size=0.2)
    assert (y_train == 0).sum() == 8
    assert (y_train == 1).sum() == 8
    assert (y_test == 0).sum() == 2
    assert (y_test == 1).sum() == 2


def test_split_data_reproducible(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    _, _, y_train1, _ = split_data(X, y, random_state=42)
    _, _, y_train2, _ = split_data(X, y, random_state=42)
    pd.testing.assert_series_equal(y_train1.reset_index(drop=True),
                                   y_train2.reset_index(drop=True))

def test_get_class_distribution_keys_and_sums(sample_df):
    X, y, _, _ = prepare_data(sample_df)
    _, _, y_train, y_test = split_data(X, y, test_size=0.2)
    dist = get_class_distribution(y_train, y_test)
    assert set(dist.keys()) == {"train", "test"}
    assert sum(dist["train"].values()) == len(y_train)
    assert sum(dist["test"].values()) == len(y_test)
