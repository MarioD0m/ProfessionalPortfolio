import streamlit as st
import pandas as pd
from data_preprocessing_c import load_and_preprocess_data as lpd
from db_conn_c import main as dbcc
from data_prep_ml_c import prepare_ml_data as dpm
from xgb_training import xgboost_training as xgbc_tr
from rf_training import random_forest_training as rfc_tr
from model_eval_c import model_evaluation as modevc
from pdf_gen_c import create_pdf_report as pdfg

def create_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.markdown("<h1 style='text-align: center; color: white;'>🚛 RouteZero: Automated ML Pipeline</h1>", unsafe_allow_html=True)
st.write("Upload your cleaned logistics dataset to execute the XGBoost vs Random Forest A/B Test.")

if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'pipeline_complete' not in st.session_state:
    st.session_state.pipeline_complete = False

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    if st.session_state.processed_file != uploaded_file.name:
        st.session_state.processed_file = uploaded_file.name
        st.session_state.pipeline_complete = False 
        
    st.success("File successfully uploaded!")

    if not st.session_state.pipeline_complete:
        
        config_container = st.empty()
        
        with config_container.container():
            st.write("Database Configuration")
            col_a, col_b = st.columns(2)
            with col_a:
                username = st.text_input("Enter the dB username:")
                password = st.text_input("Enter the dB password:", type="password")
                host = st.text_input("Enter the dB host:")
            with col_b:
                port = st.text_input("Enter the dB port:")
                db_name = st.text_input("Enter the dB name:")
                db_table = st.text_input("Enter the dB table name:")
            
            start_pipeline = st.button("Run Pipeline")

        if start_pipeline:
            
            config_container.empty()
            
            with st.spinner('Training models and computing metrics...'):
                norm_df, dcust, dprod, ddept, ford = lpd(uploaded_file)
            with st.spinner('Establishing database connection...'):
                db_info, engine, conn_ = dbcc(username, password, host, port, db_name)
                if conn_ == True:
                    st.success("Database connection successful! Preparing data for ML pipeline...")
                    X_train, y_train, X_vault, y_vault, features = dpm(db_table, engine)
                else:
                    st.error("❌ Database connection failed! Please check your credentials and host.")
                    st.stop()

            with st.spinner('Training XGBoost model...'):
                xgb_df, xgb_metrics, best_xgb, xgb_predictions = xgbc_tr(X_train, y_train, X_vault, y_vault)
                st.success("XGBoost training complete!")
            with st.spinner('Training Random Forest model...'):
                rf_df, rf_metrics, best_rf, rf_predictions = rfc_tr(X_train, y_train, X_vault, y_vault)
                st.success("Random Forest training complete!")
            with st.spinner('Evaluating models and selecting champion...'):
                t_test, champion, reason, importance_df = modevc(xgb_metrics, rf_metrics, best_xgb, best_rf, xgb_predictions, 
                                                        rf_predictions, y_vault, features)
                st.success(f"Model evaluation complete! Champion Model: {champion}")
            with st.spinner('Generating PDF report...'):
                models_df = pd.DataFrame({
                    'model': ['XGBoost', 'Random Forest'],
                    'mae': [xgb_df['mae'].iloc[0], rf_df['mae'].iloc[0]],
                    'rmse': [xgb_df['rmse'].iloc[0], rf_df['rmse'].iloc[0]],
                    'r2': [xgb_df['r2'].iloc[0], rf_df['r2'].iloc[0]],
                    'training_time': [xgb_df['training_time'].iloc[0], rf_df['training_time'].iloc[0]],
                    'best_params': [xgb_df['best_params'].iloc[0], rf_df['best_params'].iloc[0]]
                }).set_index('model')
                
                pdf_file_path = pdfg(t_test, champion, reason, importance_df, models_df)

                st.session_state.dim_customers_csv = create_csv(dcust)
                st.session_state.dim_products_csv = create_csv(dprod)
                st.session_state.dim_departments_csv = create_csv(ddept)
                st.session_state.fact_orders_csv = create_csv(ford)
                st.session_state.pdf_file_path = pdf_file_path
                
                st.session_state.pipeline_complete = True
                st.rerun() 

    if st.session_state.pipeline_complete:
        st.success("✅ Pipeline Complete! The Champion Model has been evaluated.")
            
        st.write("Generated CSV Files")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label="💾 Download dim_customers.csv", data=st.session_state.dim_customers_csv, 
                            file_name="dim_customers.csv", mime="text/csv")
            st.download_button(label="💾 Download dim_departments.csv", data=st.session_state.dim_departments_csv, 
                            file_name="dim_departments.csv", mime="text/csv")
        with col2:
            st.download_button(label="💾 Download dim_products.csv", data=st.session_state.dim_products_csv, 
                            file_name="dim_products.csv", mime="text/csv")
            st.download_button(label="💾 Download fact_orders.csv", data=st.session_state.fact_orders_csv, 
                            file_name="fact_orders.csv", mime="text/csv")

        st.write("Executive Report")
        with open(st.session_state.pdf_file_path, "rb") as pdf_file:
            st.download_button(
                label="💾 Download Executive PDF Report",
                data=pdf_file,
                file_name="RouteZero_Executive_Report.pdf",
                mime="application/pdf",
                type="primary"
            )