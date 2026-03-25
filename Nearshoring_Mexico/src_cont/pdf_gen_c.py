from fpdf import FPDF
import pandas as pd

def create_pdf_report(t_test, champion, reason, importance_df, models_df, output_filename="RouteZero_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="RouteZero: MLOps A/B Testing Report", ln=True, align='C')
    pdf.ln(10)
    
    # Decision Logic
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Decision Logic & Champion Selection:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"-> {t_test}", ln=True)
    pdf.multi_cell(0, 10, txt=f"-> The Champion Model selected is: {champion}.\n" 
                              f"-> This decision was based on: {reason}")
    pdf.ln(5)
    
    # Model Performance Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Model Performance Metrics:", ln=True)
    pdf.set_font("Arial", size=11)
    
    for model, stats in models_df.iterrows():
        pdf.multi_cell(0, 10, txt=f"-> {model}: \n"
                                  f"    ---> MAE = {stats['mae']:.2f} | RMSE = {stats['rmse']:.2f} | R^2 = {stats['r2']:.2f} | Time = {stats['training_time']}\n"
                                  f"    ---> Best parameters = {stats['best_params']}")
        
    pdf.cell(200, 10, txt="3. Feature Importance:", ln=True)  # Add space
    pdf.set_font("Arial", size=11)
    
    for col in importance_df.columns:
        pdf.cell(60, 10, col, border=1)
    pdf.ln()

    for index, row in importance_df.iterrows():
        for datum in row.values:
            pdf.cell(60, 10, str(datum), border=1)
        pdf.ln()

    pdf.output(output_filename)
    return output_filename