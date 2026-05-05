import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    return df
    
def validate_dataset(df):
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if "SalePrice" in df.columns and df["SalePrice"].isnull().all():
        raise ValueError("SalePrice column has all null values")
        
    print(f"Dataset validated : {df.shape[0]} rows and {df.shape[1]} columns.")
    
def split_data(df):
    df = df.copy()
    X = df.drop("SalePrice", axis = 1)
    y = df["SalePrice"]
    return X, y

def split_train_test(X, y):
    X = X.copy()
    y = y.copy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test

def save_processed_data(path, df):
    df_saved = df.copy()
    df_saved.to_csv(path, index = False)

def get_data_info(df):
    print("*"*40)
    print(f"Shape of data          : {df.shape}")
    print(f"columns count          : {len(df.columns)}")
    print(f"Null Values            : {df.isnull().sum().sum()}")
    print(f"Duplicates count       : {df.duplicated().sum()}")
    print(f"Dtypes                 : \n {df.dtypes.value_counts()}")
    print("*"*40)
