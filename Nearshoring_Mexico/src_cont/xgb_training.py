import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import timedelta

def xgboost_training(X_train_cv, y_train_cv, X_vault, y_vault):
    print('\nINITIATING XGBOOST TRAINING ...')
    xgb_start_time = time.perf_counter()

    # Define the Grid parameters for XGBoost
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.7, 1.0]
    }

    # Initialize Random Forest regression model with all available CPU
    print('\n -> Executing XGBoost training ...')
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')

    # 5-Fold Cross Validation for XGBoost hyperparameters
    print(' -> Executing a 5-fold Cross Validation for XGBoost hyperparameters ...')
    grid_search_xgb = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid_xgb,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    # Hyperparameter search and model training process
    print(' -> Executing hyperparameter search and model training ...')
    grid_search_xgb.fit(X_train_cv, y_train_cv)
    best_xgb = grid_search_xgb.best_estimator_

    # Evaluate model using Vault data
    print(' -> Evaluating model using Vault data ...')
    xgb_predictions = best_xgb.predict(X_vault)
    xgb_mae = mean_absolute_error(y_vault, xgb_predictions)
    xgb_rmse = np.sqrt(mean_squared_error(y_vault, xgb_predictions))
    xgb_r2 = r2_score(y_vault, xgb_predictions)

    print('\n XGBoost model training complete. ')
    xgb_end_time = time.perf_counter()
    # Calculate the duration of the training process
    xgb_elapsed_time_secs = xgb_end_time - xgb_start_time
    xgb_final_time = timedelta(seconds=round(xgb_elapsed_time_secs))

    # Compile results into a dictionary for reporting and future reference
    xgb_metrics = {'model': 'XGBoost', 'best_params': grid_search_xgb.best_params_, 'mae': xgb_mae,
                  'rmse': xgb_rmse, 'r2': xgb_r2, 'training_time': str(xgb_final_time)}
    xgb_df = pd.DataFrame([xgb_metrics])
    print(f"\n XGBoost Metrics: {xgb_metrics}")

    return xgb_df, xgb_metrics, best_xgb, xgb_predictions