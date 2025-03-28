import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

# Suppress warnings
warnings.filterwarnings("ignore")

def convert_minutes(val):
    """Convert minutes to categorical output"""
    if val > 10:
        return 1
    else:
        return 0

def load_and_prepare_data(gameweek):
    """Load and prepare data for the specified gameweek"""
    # Load training data
    train = pd.read_csv("cleaned_dataset/cleaned_previous_seasons.csv", index_col=0)
    
    # Load current season data
    old_gameweek_cleaned = []
    for i in range(1, gameweek):
        old_gameweek_cleaned.append(pd.read_csv(f"cleaned_dataset/2023-24/GW{i}.csv"))
    old_gameweeks = pd.concat(old_gameweek_cleaned)[train.columns]
    train = pd.concat([train, old_gameweeks])
    
    # Load test data
    test = pd.read_csv(f"cleaned_dataset/2023-24/GW{gameweek}.csv", index_col=0)
    
    return train, test

def prepare_features(train, test):
    """Prepare features for model training"""
    # Create unique index
    train["index"] = train["name"] + train["kickoff_time"].astype("str")
    test["index"] = test["name"] + test["kickoff_time"].astype("str")
    
    # Set index
    train = train.set_index("index")
    test = test.set_index("index")
    
    # Convert minutes to categorical
    train["minutes"] = train["minutes"].apply(convert_minutes)
    
    # Prepare target
    target = train[["minutes", "GW", "position"]]
    
    # Drop unnecessary columns
    train.drop(["total_points", "minutes"], axis=1, inplace=True)
    test.drop(["total_points", "minutes"], axis=1, inplace=True)
    
    # Convert categorical columns
    for col in train.columns:
        if train[col].dtype == "object":
            if col not in ["team", "name", "position"]:
                train[col] = pd.factorize(train[col])[0]
                test[col] = pd.factorize(test[col])[0]
    
    # Convert boolean columns
    train["was_home"] = train["was_home"].replace({True: 0, False: 1})
    test["was_home"] = test["was_home"].replace({True: 0, False: 1})
    
    return train, test, target

def train_model(train, test, target):
    """Train the CatBoost model"""
    # Split data
    x, val, y, y_val = train_test_split(
        train.drop(["name", "team"], axis=1),
        target["minutes"],
        test_size=0.1,
        random_state=0,
    )
    
    # Model parameters
    params = {
        'colsample_bylevel': 0.8070621518153563,
        'learning_rate': 0.04765984972709895,
        'max_depth': 7,
        'reg_lambda': 5,
        'scale_pos_weight': 2.5,
        'subsample': 0.6794390204583894
    }
    
    # Initialize and train model
    model = CatBoostClassifier(
        **params,
        cat_features=["position"],
        random_state=0,
        early_stopping_rounds=500,
        use_best_model=True,
        verbose=500,
        n_estimators=10000
    )
    
    # Fit model
    model.fit(x, y, eval_set=[(val, y_val)])
    
    return model, val, y_val

def evaluate_model(model, val, y_val):
    """Evaluate model performance"""
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(model.predict(val), y_val))
    
    # Print accuracy score
    print(f"\nAccuracy Score: {accuracy_score(model.predict(val), y_val)}")
    
    # Print F1 score
    print(f"F1 Score: {f1_score(model.predict(val), y_val)}")
    
    # Get feature importance
    feature_importance = pd.DataFrame(
        {"column": val.columns, "imp": model.feature_importances_}
    ).sort_values("imp", ascending=False)
    
    return feature_importance

def main():
    # Set gameweek
    gameweek = 10
    
    # Create predictions directory
    path = f"predicted_dataset/GW{gameweek}"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train, test = load_and_prepare_data(gameweek)
    train, test, target = prepare_features(train, test)
    
    # Train model
    print("\nTraining model...")
    model, val, y_val = train_model(train, test, target)
    
    # Evaluate model
    print("\nEvaluating model...")
    feature_importance = evaluate_model(model, val, y_val)
    
    # Save feature importance
    feature_importance.to_csv(f"{path}/feature_importance.csv")
    print(f"\nFeature importance saved to {path}/feature_importance.csv")

if __name__ == "__main__":
    main() 