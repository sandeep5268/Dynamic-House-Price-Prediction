import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import  Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.features.build_features import house_price_cleaner, frontage_cleaning
from src.features.build_features import clipper, feature_engineer, build_preprocessor
from src.data.make_dataset import load_dataset
from src.config import LOAD_RAW_DATA, LOAD_PROCESSED_DATA, EXPERIMENTS_DIR, RESULTS_DIR
from src.data.make_dataset import split_data, split_train_test, save_processed_data, get_data_info, validate_dataset

def make_pipeline(preprocessor, clipper, feature_engineer, model):
    return Pipeline([
        ('clipper' , clipper),
        ('feature_engineering', feature_engineer),
        ('processor' , preprocessor),
        ('model', model)
    ])

def build_models(preprocessor, clipper, feature_engineer):
    models_config ={
        # Ridge
        "Ridge" : {
            "pipeline" : make_pipeline(preprocessor, clipper, feature_engineer, Ridge()),
            "params" : {
                "model__alpha" : [0.01, 0.1, 1, 10]
            }
        },
        # Lasso
        "Lasso" : {
            "pipeline" : make_pipeline(preprocessor, clipper, feature_engineer, Lasso()),
            "params" : {
                "model__alpha" : [0.001, 0.01, 0.1, 1]
            }
        },
        #Gradient Regressor
        "GradientBoost" : {
            "pipeline" : make_pipeline(preprocessor, clipper, feature_engineer, GradientBoostingRegressor(random_state=42)),
            "params" : {
                "model__n_estimators" : [100, 200],
                "model__learning_rate" : [0.01, 0.1],
                "model__max_depth" : [3, 5]
            }
        },
        # XGBOOST regressor
        "XGBoost" : {
            "pipeline" :  make_pipeline(preprocessor, clipper, feature_engineer, XGBRegressor(random_state=42)),
            "params" : {
                "model__n_estimators" : [100, 200],
                "model__learning_rate" : [0.01, 0.1],
                "model__max_depth" : [3, 6],
            }
        },
        # Support Vector Regressor
        "SVR" : {
            "pipeline" :  make_pipeline(preprocessor, clipper, feature_engineer, SVR()),
            "params" : {
                "model__C" : [0.1, 1, 10],
                "model__epsilon" : [0.1, 0.2],
                "model__kernel" : ['rbf']
            }
        }, 
        # Elastic Net
        "ElasticNet" : {
            "pipeline" :  make_pipeline(preprocessor, clipper, feature_engineer, ElasticNet(max_iter=10000,tol = 1e-5)),
            "params" : {
                "model__alpha" : [0.0005, 0.0001, 0.001, 0.005, 0.01],
                "model__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
            }
        },    
    }
    
    
    models = {
        name : GridSearchCV(
            config["pipeline"],
            config["params"],
            cv = KFold(n_splits = 5, shuffle = True, random_state = 42),
            scoring = "neg_root_mean_squared_error",
            n_jobs=-1
        )
        for name, config in models_config.items()
    }
    return models

def evaluate_model(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_score = float("inf")
    best_model_name = None 
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = -model.best_score_
        
        results.append({
            "Model" : name,
            "R2" : round(r2, 4),
            "RMSE" : round(rmse, 4),
            "MAE" : round(mae, 4)
        })

        print(f"{name} Model R2 Score: {r2}")
        print(f"{name} Model cross value rmse: {rmse}")
        print(f"{name} Model Mean Absolute Error: {mae}\n")

    
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_model_name = name
    return best_model, best_model_name, results


def save_model(best_model, path):
    joblib.dump(best_model, path)
    
def save_results(results, best_model_name, results_dir):

    df = pd.DataFrame(results)
    df.to_csv(results_dir/ "model_comparison.csv", index = False)

    best = df[df["Model"] == best_model_name].iloc[0]
    with open(results_dir/ "best_model_summary.txt", "w") as f:
        f.write(f"Best Model       : {best['Model']}\n")
        f.write(f"R2 Score         : {best['R2']}\n")
        f.write(f"RMSE Score       : {best['RMSE']}\n")
        f.write(f"MAE Score        : {best['MAE']}\n")
        
    print("Results saved to reports/results/")

def main():
    # Load raw data
    data = load_dataset(LOAD_RAW_DATA / "training_dataset.csv")

    # validate data
    validate_dataset(data)  

    # transform the target variable
    data["SalePrice"] = np.log1p(data["SalePrice"])

    # initial data info
    print("Initial data info:")
    get_data_info(data)

    # Splitting the data
    X, y = split_data(data)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Get data info before cleaning and splitting
    print("x train info before cleaning:")
    get_data_info(X_train)
    print("x test info before cleaning: ")
    get_data_info(X_test)

    # Clean BOTH train and test splits — must be done before any pipeline step
    X_train = house_price_cleaner(X_train)
    X_train = frontage_cleaning(X_train)
    X_test = house_price_cleaner(X_test)
    X_test = frontage_cleaning(X_test)
    
    # Get data info after cleaning
    print("x train info after cleaning:")
    get_data_info(X_train)

    # Build preprocessor on a feature-engineered sample so column names are correct
    sample_data = feature_engineer.transform(clipper.transform(X_train))
    preprocessor = build_preprocessor(sample_data)

    # Build all model pipelines
    model_pipelines = build_models(preprocessor, clipper, feature_engineer)
    print()
    
    # Saving the Processed data
    save_processed_data(LOAD_PROCESSED_DATA / "X_train_processed.csv", X_train)
    save_processed_data(LOAD_PROCESSED_DATA / "y_train.csv", y_train)
    save_processed_data(LOAD_PROCESSED_DATA / "X_test_processed.csv", X_test)
    save_processed_data(LOAD_PROCESSED_DATA / "y_test.csv", y_test)
    
    # Evaluating the model
    best_model, best_model_name, results = evaluate_model(model_pipelines, X_train, y_train, X_test, y_test)

    # Save the best model
    save_model(best_model, EXPERIMENTS_DIR / "best_model.pkl")
    print("Best model saved to experiments/best_model.pkl")

    # Save the results
    save_results(results, best_model_name, RESULTS_DIR)


if __name__ == "__main__":
    main()