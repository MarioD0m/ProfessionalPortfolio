from scipy import stats
import numpy as np
import pandas as pd

def model_evaluation(xgb_metrics, rf_metrics, best_xgb, best_rf, xgb_predictions, rf_predictions, y_vault, features):
    # In order to stablish if there is a significant difference between the models, and Offline A/B Test
    # is needed. A paired Student's t-test on the absolute errors of every single prediction in the Vault
    #is made to determine if the difference is real, or is it just random statistical noise.

    # Calculate the absolute error for every single prediction in the Vault
    rf_errors = np.abs(y_vault - rf_predictions)
    xgb_errors = np.abs(y_vault - xgb_predictions)

    # Run the Paired T-Test
    t_stat, p_value = stats.ttest_rel(rf_errors, xgb_errors)

    # Business Logic Evaluation
    alpha = 0.05
    t_test = ''
    if p_value < alpha:
        t_test = f'The p-value of {p_value:.4f} is statistically significant'
    else:
        t_test = f'The p-value of {p_value:.4f} is not statistically significant'

    winner = None
    winning_reason = ''
    error_tolerance = 0.01

    # RANK 1: RMSE (The "Risk" Metric)
    # If the difference in RMSE is larger than our tolerance, the lowest RMSE wins immediately.
    rmse_diff = abs(rf_metrics['rmse'] - xgb_metrics['rmse'])

    if rmse_diff > error_tolerance:
        if rf_metrics['rmse'] < xgb_metrics['rmse']:
            winner, winning_reason = 'Random Forest', f'Superior RMSE (Margin: {rmse_diff:.3f} days)'
        else:
            winner, winning_reason = 'XGBoost', f'Superior RMSE (Margin: {rmse_diff:.3f} days)'

    else:
        pass

        # RANK 2: MAE (The "Translation" Metric)
        mae_diff = abs(rf_metrics['mae'] - xgb_metrics['mae'])

        if mae_diff > error_tolerance:
            if rf_metrics['mae'] < xgb_metrics['mae']:
                winner, winning_reason = 'Random Forest', f'Superior MAE (Margin: {mae_diff:.3f} days)'
            else:
                winner, winning_reason = 'XGBoost', f'Superior MAE (Margin: {mae_diff:.3f} days)'

        else:
            pass

            # RANK 3: TIME (The "Engineering" Tie-Breaker)
            # If accuracy is identical, the fastest model wins to save cloud computing costs.
            if rf_metrics['training_time'] < xgb_metrics['training_time']:
                winner, winning_reason = 'Random Forest', 'Superior Training Time (Tie-breaker)'
            else:
                winner, winning_reason = 'XGBoost', 'Superior Training Time (Tie-breaker)'

    if winner == 'Random Forest':
        best_model = best_rf
    else:
        best_model = best_xgb

    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    print(importance_df.to_string(index=False))

    return t_test, winner, winning_reason, importance_df