# src/data_preprocessing.py
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path, file_name):
    print("INITIATING DATA ENGINEERING PIPELINE ...")

    print(f"\n -> Loading data from local file: {file_name}...")
    
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(f'{file_path}{file_name}.csv', encoding='latin1')

    except FileNotFoundError:
        print(f"ERROR: Could not find {file_name}. Please check your data folder.")
        return None, None, None, None, None
    
    print(f' -> Uploading file: {file_name}.csv, [{df.shape[0]} records and {df.shape[1]} features] ...')

    #Filter for records of Mexico as the order country
    df_mexico = df[df['Order Country'] == 'México'].copy()
    print(f" -> Filtering records for Mexico as the order country [{len(df_mexico)} records] ...")

    #Standardise the column names
    print(' -> Standardizing feature names and values ...')
    df_mexico.columns = [c.strip().lower().replace(' ', '_') for c in df_mexico.columns]

    # Clean and standardize zipcode by removing decimal points and ensuring 5-digit format
    df_mexico['customer_zipcode'] = df_mexico['customer_zipcode'].astype(str).str.replace('.0','').str.zfill(5)

    # Convert date columns to datetime and extract year, month, day for future time-based analysis
    df_mexico['order_date_(dateorders)'] = pd.to_datetime(df_mexico['order_date_(dateorders)'], errors='coerce')
    df_mexico['shipping_date_(dateorders)'] = pd.to_datetime(df_mexico['shipping_date_(dateorders)'], errors='coerce')
    df_mexico['order_year'] = df_mexico['order_date_(dateorders)'].dt.year
    df_mexico['order_month'] = df_mexico['order_date_(dateorders)'].dt.month
    df_mexico['order_day_of_week'] = df_mexico['order_date_(dateorders)'].dt.dayofweek

    # Create a binary feature to indicate if the order was placed on a weekend (1) or weekday (0)
    df_mexico['is_weekend'] = np.where(df_mexico['order_day_of_week'] >= 5, 1, 0)

    #Drop columns with more that 80% of null values as they won't contribute to the analysis and may introduce bias
    print(' -> Droping features with less than 80% of data integrity ...')
    features_with_nulls = pd.DataFrame(df_mexico.isnull().sum()[df_mexico.isnull().sum() > 0], columns = ['Null_count'])
    features_to_drop = (features_with_nulls[features_with_nulls.iloc[:, 0]/df_mexico.shape[0]>=0.8]).index
    df_mexico.drop(columns=features_to_drop, inplace=True)

    df_mexico[['product_price', 'order_item_total', 'order_profit_per_order']] = df_mexico[['product_price', 
                                            'order_item_total', 'order_profit_per_order']].round(2)

    # Define dtypes for dimension and fact tables, assuming df_mexico is now cleaned
    # These dtypes ensure correct types for newly created DFs
    customer_dtypes = {'customer_id': 'int64', 'customer_segment': 'str','customer_city': 'str',
                        'customer_state': 'str','customer_country': 'str','customer_zipcode': 'str'}
    product_dtypes = {'product_card_id': 'int64','product_category_id': 'int64',
                        'category_name': 'str', 'product_name': 'str', 'product_price': 'float64'}
    department_dtypes = {'department_id': 'int64', 'department_name': 'str'}
    order_dtypes = {'order_id': 'int64', 'order_item_id': 'int64', 'customer_id': 'int64',
                    'product_card_id': 'int64', 'department_id': 'int', 'shipping_mode': 'str',
                    'order_date_(dateorders)': 'datetime64[us]', 'shipping_date_(dateorders)': 'datetime64[us]',
                    'days_for_shipping_(real)': 'int64', 'days_for_shipment_(scheduled)': 'int64',
                    'order_item_total': 'float64', 'order_profit_per_order': 'float64', 'order_year': 'int64',
                    'order_month': 'int64', 'order_day_of_week': 'int64', 'is_weekend': 'int64'}

    # Normalize data into 3 Dimension tables and 1 Fact table
    print(' -> Normalizing raw file into Dimension and Fact tables ...')
    df_dim_customers = pd.DataFrame(df_mexico[[ #First Dim table
        'customer_id', 'customer_segment','customer_city','customer_state',
        'customer_country','customer_zipcode']]).drop_duplicates().astype(customer_dtypes)

    df_dim_products = pd.DataFrame(df_mexico[[ #Second Dim table
        'product_card_id','product_category_id', 'category_name','product_name', 'product_price']]
    ).drop_duplicates().astype(product_dtypes)

    df_dim_departments = pd.DataFrame(df_mexico[[ #Third Dim table
        'department_id', 'department_name']]).drop_duplicates().astype(department_dtypes)

    df_fact_orders = pd.DataFrame(df_mexico[[ #First Fact table
            'order_id', 'order_item_id', 'customer_id', 'product_card_id', 'department_id',
            'shipping_mode', 'order_date_(dateorders)', 'shipping_date_(dateorders)',
            'days_for_shipping_(real)', 'days_for_shipment_(scheduled)','order_item_total',
            'order_profit_per_order', 'order_year', 'order_month', 'order_day_of_week', 
            'is_weekend']]).drop_duplicates().astype(order_dtypes)

    #Cap extreme profits/losses at the 1st and 99th percentiles to stabilize future regression
    print(' -> Handling outliers by capping extreme values (range: 1-99%) ...')
    df_fact_orders['order_profit_per_order'] = np.clip(df_fact_orders['order_profit_per_order'],
        df_fact_orders['order_profit_per_order'].quantile(0.01),
        df_fact_orders['order_profit_per_order'].quantile(0.99))
    df_fact_orders.rename(columns = {'order_profit_per_order': 'order_profit_capped'}, inplace=True)
    df_fact_orders[['order_profit_capped']] = df_fact_orders[['order_profit_capped']].round(2)

    # Create new features: item quantity and profit margin per order item
    df_fact_orders['item_quantity'] = np.where(df_mexico['product_price'] > 0, 
                    df_fact_orders['order_item_total'] / df_mexico['product_price'], 1).astype(int).round(2)
    df_fact_orders['profit_margin'] = np.where(df_fact_orders['order_item_total'] > 0, 
                    df_fact_orders['order_profit_capped'] / df_fact_orders['order_item_total'], 0).astype(float).round(2)
    df_fact_orders['delivery_variance_days'] = df_fact_orders['days_for_shipping_(real)'] - df_fact_orders['days_for_shipment_(scheduled)'].astype(int)

    # Exporting dataframes into CSV files for SQL handling
    print(' -> Exporting Dimension and Fact tables to designated file path:')
    print(f' ---> (D) Customers: {file_path}dim_customers.csv')
    print(f' ---> (D) Products: {file_path}dim_products.csv')
    print(f' ---> (D) Departments: {file_path}dim_departments.csv')
    print(f' ---> (F) Orders: {file_path}fact_orders.csv')

    df_dim_customers.to_csv(f"{file_path}dim_customers.csv", index=False)
    df_dim_products.to_csv(f"{file_path}dim_products.csv", index=False)
    df_dim_departments.to_csv(f"{file_path}dim_departments.csv", index=False)
    df_fact_orders.to_csv(f"{file_path}fact_orders.csv", index=False)

    print("\n Pipeline Complete. Normalized tables exported successfully.")

    return df_mexico, df_dim_customers, df_dim_products, df_dim_departments, df_fact_orders