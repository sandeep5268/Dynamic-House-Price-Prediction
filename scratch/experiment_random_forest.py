import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from src.config import LOAD_PROCESSED_DATA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    print("Experimenting thee Random Forest Regressor")
    
    X_train = pd.read_csv(LOAD_PROCESSED_DATA / "X_train_processed.csv")
    y_train = pd.read_csv(LOAD_PROCESSED_DATA / "y_train.csv")
    X_test = pd.read_csv(LOAD_PROCESSED_DATA / "X_test_processed.csv")
    y_test = pd.read_csv(LOAD_PROCESSED_DATA / "y_test.csv")
    
    print(f"Loaded processed data. Training on {len(X_train)} rows ")
    
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("Results:")
    print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
    print(f"RMSE : {root_mean_squared_error(y_test, predictions):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, predictions):.4f}")