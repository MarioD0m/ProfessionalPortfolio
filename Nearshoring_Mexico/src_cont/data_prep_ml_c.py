# src/data_preparation_ml.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split as tts

def prepare_ml_data(db_table, engine):

    df = pd.read_sql(f"SELECT * FROM {db_table};", engine)

    # Feature engineering (One-hot encoding)
    encoded_df = pd.get_dummies(df, columns=['shipping_mode', 'customer_segment','customer_city','customer_state', 
        'category_name', 'department_name'], drop_first=True)

    # Split the data into training and test (vault) sets
    X = encoded_df.drop('delivery_variance_days', axis=1)
    y = encoded_df['delivery_variance_days']
    X_train_cv, X_vault, y_train_cv, y_vault = tts(X,y, test_size=0.20, random_state=16)

    # Extract feature names for later use in model evaluation and interpretation
    features = X_train_cv.columns.tolist()

    return X_train_cv, y_train_cv, X_vault, y_vault, features