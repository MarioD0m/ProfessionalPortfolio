# src/data_preparation_ml.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split as tts

def prepare_ml_data(db_table, engine):
    print("\nINITIATING DATA PREPARATION FOR ML PIPELINE ...")
    print("\n -> Loading data from MySQL View...")

    df = pd.read_sql(f"SELECT * FROM {db_table};", engine)
    print(f' -> Uploading View: {db_table}, [{df.shape[0]} records and {df.shape[1]} features] ...')

    # Feature engineering (One-hot encoding)
    print(' -> Encoding categorical features ...')
    encoded_df = pd.get_dummies(df, columns=['shipping_mode', 'customer_segment','customer_city','customer_state', 
        'category_name', 'department_name'], drop_first=True)

    # Split the data into training and test (vault) sets
    print(' -> Splitting data into training and test sets ...')
    X = encoded_df.drop('delivery_variance_days', axis=1)
    y = encoded_df['delivery_variance_days']
    X_train_cv, X_vault, y_train_cv, y_vault = tts(X,y, test_size=0.20, random_state=16)
    print(f'\n Data Split Complete. Vault set archived.')
    print(f' ---> Training/CV Set: {X_train_cv.shape[0]} records. | Vault Set: {X_vault.shape[0]} records.')    

    # Extract feature names for later use in model evaluation and interpretation
    features = X_train_cv.columns.tolist()

    return X_train_cv, y_train_cv, X_vault, y_vault, features
