import pytest
import pandas as pd
from src.data.make_dataset import (
    validate_dataset, split_data, split_train_test, save_processed_data
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "SalePrice": [200000, 150000, 300000, 250000, 180000],
        "GrLivArea": [1500, 1200, 2000, 1800, 1300],
        "OverallQual": [7, 5, 8, 7, 6],
        "Neighborhood": ["NAmes", "CollgCr", "OldTown", "NAmes", "CollgCr"]
    })
    

def test_validate_dataset_passes(sample_df):
    validate_dataset(sample_df)

def test_validate_dataset_raises_on_empty():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match = "Dataset is empty"):
        validate_dataset(empty_df)
    
def test_split_data_returns_correct_shapes(sample_df):
    X, y = split_data(sample_df)
    assert "SalePrice" not in X.columns
    assert len(y) == len(sample_df)
    
def test_split_train_test_sizes(sample_df):
    X, y = split_data(sample_df)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) > len(X_test)
    
def test_save_processed_data(tmp_path, sample_df):
    
    output_path = tmp_path/ "test_output.csv"
    save_processed_data(output_path, sample_df)
    assert output_path.exists()
    
    loaded = pd.read_csv(output_path)
    assert loaded.shape == sample_df.shape
    