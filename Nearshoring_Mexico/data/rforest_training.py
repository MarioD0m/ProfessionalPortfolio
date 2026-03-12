import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import timedelta
import joblib

def random_forest_training(X_train_cv, y_train_cv, X_vault, y_vault, file_path):
    print('\nINITIATING RANDOM FOREST TRAINING ...')
    rf_start_time = time.perf_counter()

    # Define the Grid parameters for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'max_features': [0.33, 1.0]
    }

    # Initialize Random Forest regression model with all available CPU
    print('\n -> Executing Random Forest training ...')
    rf_base = RandomForestRegressor(random_state=16, n_jobs=-1)

    # Initialize a 5-Fold Cross Validation for RandomForest hyperparameters
    print(' -> Executing a 5-fold Cross Validation for RF hyperparameters ...')
    grid_search_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid_rf,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    # Hyperparameter search and model training process
    print(' -> Executing hyperparameter search and model training ...')
    grid_search_rf.fit(X_train_cv, y_train_cv)
    best_rf = grid_search_rf.best_estimator_

    # Evaluate model using Vault data
    print(' -> Evaluating model using Vault data ...')
    rf_predictions = best_rf.predict(X_vault)
    rf_mae = mean_absolute_error(y_vault, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_vault, rf_predictions))
    rf_r2 = r2_score(y_vault, rf_predictions)

    print('\n Random Forest model training complete. ')

    rf_end_time = time.perf_counter()
    # Calculate the duration
    rf_elapsed_time_secs = rf_end_time - rf_start_time
    rf_final_time = timedelta(seconds=round(rf_elapsed_time_secs))

    # Compile results into a dictionary for reporting and future reference
    rf_metrics = {'model': 'Random Forest', 'best_params': grid_search_rf.best_params_, 'mae': rf_mae,
                  'rmse': rf_rmse, 'r2': rf_r2, 'training_time': str(rf_final_time)}
    rf_df = pd.DataFrame([rf_metrics])


    # Save the best Random Forest model to a file for future use
    joblib.dump(best_rf, file_path + 'random_forest_model.joblib')

    return rf_df, rf_metrics, best_rf, rf_predictions