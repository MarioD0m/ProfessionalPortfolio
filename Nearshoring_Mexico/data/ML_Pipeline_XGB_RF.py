import pandas as pd
import data_preprocessing as dp
import data_preparation_ml as dpm
import db_connection as dbc
import xgboost_training as xgb_tr
import rforest_training as rf_tr
import model_evaluation as modev

# User inputs for dataset file path and name
file_path = str(input("Enter the file path for the dataset (e.g., /Users/yourname/Desktop/): "))
file_name = str(input("Enter the file name for the dataset (without .csv extension, e.g., DataCoSupplyChainDataset): "))

# Step 1: Load and preprocess the data
norm_df, dcust, dprod, ddept, ford = dp.load_and_preprocess_data(file_path, file_name)

# Step 2: Establish database connection and prepare data for ML pipeline
db_info, db_table, engine, conn_ = dbc.main()

# Step 3: If connection is successful, prepare the training and vault datasets
if conn_ == True:
    X_train, y_train, X_vault, y_vault, features = dpm.prepare_ml_data(db_table, engine)

# Step 4: Train XGBoost and Random Forest models
xgb_df, xgb_metrics, best_xgb, xgb_predictions = xgb_tr.xgboost_training(X_train, y_train, X_vault, y_vault, file_path)
rf_df, rf_metrics, best_rf, rf_predictions = rf_tr.random_forest_training(X_train, y_train, X_vault, y_vault, file_path)

# Step 5: Evaluate models and select champion
champion, reason, importance_df = modev.model_evaluation(xgb_metrics, rf_metrics, best_xgb, best_rf, xgb_predictions, rf_predictions, y_vault, features)

# Step 6: Compile and display model performance metrics
models_df = pd.concat([xgb_df, rf_df], ignore_index=True)
print("\n -> Model Performance Metrics:")
print(models_df)