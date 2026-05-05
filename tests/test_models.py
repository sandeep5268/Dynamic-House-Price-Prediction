import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.models.train import evaluate_model, save_model

@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1" : np.random.randint(100, size=100),
        "feat2" : np.random.randint(100, size=100)
    })
    y = pd.Series(np.random.randint(100, size=100))
    return X, y
    
@pytest.fixture
def dummy_models():
    pipeline = Pipeline([
        ("model" , Ridge())
    ])
    
    gs = GridSearchCV(
        pipeline, 
        {
            "model__alpha" : [1.0]
        },
        cv = 2,
        scoring = "neg_root_mean_squared_error"
    )
    
    return {"Ridge" : gs}


def test_evaluate_model_returns_best_model(dummy_models, simple_data):
    X, y = simple_data
    
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]
    
    best_model, best_name, results = evaluate_model(dummy_models, 
                                                    X_train, y_train, X_test, y_test)
    
    assert best_model is not None
    assert best_name == "Ridge"
    assert len(results)  == 1
    assert "R2" in results[0]
    assert "RMSE" in results[0]
    
def test_save_model_creates_file(tmp_path, dummy_models, simple_data):
    X, y = simple_data
    model = dummy_models["Ridge"]
    model.fit(X, y)
    
    output = tmp_path/ "model.pkl"
    save_model(model, output)
    
    assert output.exists()  