import pytest
import numpy as np
import pandas as pd
from src.features.build_features import (
    house_price_cleaner, frontage_cleaning,
    clip_outliers, house_price_feature_engineering
)

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Id": [1, 2, 3],
        "SalePrice": [200000, 150000, 300000],
        "Alley": [None, None, None],
        "PoolQC": [None, None, None],
        "Fence": [None, None, None],
        "MiscFeature": [None, None, None],
        "MasVnrType": [None, "BrkFace", None],
        "MasVnrArea": [None, 100.0, None],
        "GarageYrBlt": [None, 2005.0, 2010.0],
        "MSZoning": [None, "RL", "RL"],
        "Neighborhood": ["NAmes", "CollgCr", "NAmes"],
        "LotFrontage": [None, 70.0, 60.0],
        
        # Columns for the feature engineering test below
        "TotalBsmtSF": [800.0, 1000.0, 900.0],
        "1stFlrSF": [1000.0, 1200.0, 1100.0],
        "2ndFlrSF": [0.0, 0.0, 0.0],
        "FullBath": [2, 2, 1],
        "HalfBath": [1, 0, 0],
        "BsmtFullBath": [1, 0, 0],
        "BsmtHalfBath": [0, 0, 0],
        "YrSold": [2010, 2009, 2008],
        "YearBuilt": [1990, 2000, 1980],
        "YearRemodAdd": [2000, 2005, 1990],
        "OverallQual": [7, 8, 6],
        "GrLivArea": [1500.0, 2500.0, 1800.0],
        "LotArea": [8000.0, 12000.0, 9000.0]
    })    
    
    
def test_housue_price_cleaner_drops_columns(raw_df):
    cleaned = house_price_cleaner(raw_df)
    for col in ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']:
        assert col not in cleaned.columns
   
        
def test_house_price_cleaner_fills_none(raw_df):
    cleaned = house_price_cleaner(raw_df)
    assert cleaned["MasVnrType"].isnull().sum() == 0
   
    
def test_house_price_cleaner_fills_zero(raw_df):
    cleaned = house_price_cleaner(raw_df)
    assert cleaned["GarageYrBlt"].isnull().sum() == 0
  
    
def test_frontage_cleaner_fill_nulls(raw_df):
    cleaned = house_price_cleaner(raw_df)
    result = frontage_cleaning(cleaned)
    assert result['LotFrontage'].isnull().sum() == 0
    
    
def test_clip_outliers_does_not_change_shape():
    df = pd.DataFrame({
        "GrLivArea": [1000, 1200, 99999],
        "TotalBsmtSF": [800, 900, 1000],
        "GarageArea": [400, 500, 600],
        "1stFlrSF": [900, 1000, 1100],
        "LotArea": [7000, 8000, 9000]
    })
    clipped = clip_outliers(df)
    assert clipped.shape == df.shape
    assert clipped['GrLivArea'].max() <= df["GrLivArea"].max()


def test_feature_engineering_creates_new_cols(raw_df):
    cleaned = house_price_cleaner(raw_df)
    result = house_price_feature_engineering(cleaned)
    
    assert 'TotalSF' in result.columns
    assert 'TotalBathrooms' in result.columns
    assert 'HouseAge' in result.columns
    assert 'Log_GrLivArea' in result.columns