import pandas as pd
import numpy as np
import joblib
from src.features.build_features import house_price_cleaner, frontage_cleaning
from src.config import LOAD_RAW_DATA, EXPERIMENTS_DIR

def load_model(pkl_path):
    model = joblib.load(pkl_path)
    return model

def load_input_data(input_path):
    data = pd.read_csv(input_path)
    return data

def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions


def save_predictions(predictions, output_path):
    output = pd.DataFrame({
        "PredictedSalePrice" : predictions
    })
    output.to_csv(output_path, index = False)


def main():
    # Load the saved model pipeline
    model = load_model(EXPERIMENTS_DIR / "best_model.pkl")

    # Load raw test data
    data = load_input_data(LOAD_RAW_DATA / "test.csv")

    # Apply manual cleaning
    data = house_price_cleaner(data)
    data = frontage_cleaning(data)

    # Prediction
    predictions = make_predictions(model, data)

    # Reverse the log1p transform to get actual dollar prices
    actual_prices = np.expm1(predictions)

    # Save and show results
    save_predictions(actual_prices, EXPERIMENTS_DIR / "prices_predictions_new_data.csv")
    print("Sample predictions (first 10):")
    print(actual_prices[:10])


if __name__ == "__main__":
    main()