from scipy import stats
import numpy as np
import pandas as pd

def model_evaluation(xgb_metrics, rf_metrics, best_xgb, best_rf, xgb_predictions, rf_predictions, y_vault, features):
    # In order to stablish if there is a significant difference between the models, and Offline A/B Test
    # is needed. A paired Student's t-test on the absolute errors of every single prediction in the Vault
    #is made to determine if the difference is real, or is it just random statistical noise.
    print('\nINITIATING STATISTICAL A/B TESTING (PAIRED T-TEST) FOR MODEL COMPARISON ...')

    # 1. Calculate the absolute error for every single prediction in the Vault
    print('\n -> Calculating the absolute error of every prediction in the Vault for both models ...')
    rf_errors = np.abs(y_vault - rf_predictions)
    xgb_errors = np.abs(y_vault - xgb_predictions)

    # 2. Run the Paired T-Test
    print(' -> Running the Offline A/B Test ...')
    t_stat, p_value = stats.ttest_rel(rf_errors, xgb_errors)

    print(f' ---> T-Statistic: {t_stat:.4f}')
    print(f' ---> P-Value:     {p_value:.4f}')

    # Business Logic Evaluation
    print('\n Hypothesis testing complete. Evaluating business implications:')
    alpha = 0.05
    if p_value < alpha:
        print(' ---> The difference in MAE is STATISTICALLY SIGNIFICANT.')
        print(' ---> One model is mathematically superior to the other.')
    else:
        print(' ---> The difference in MAE is NOT statistically significant.')
        print(' ---> The models are tied in accuracy. Speed and computational cost should dictate the winner.')

    print('\nINITIATING AUTOMATED MLOPS MODEL SELECTION PIPELINE ... \n')

    winner = None
    winning_reason = ''
    error_tolerance = 0.01

    # RANK 1: RMSE (The "Risk" Metric)
    # If the difference in RMSE is larger than our tolerance, the lowest RMSE wins immediately.
    print('\n -> Evaluating metrics difference for champion selection ...')
    rmse_diff = abs(rf_metrics['rmse'] - xgb_metrics['rmse'])

    if rmse_diff > error_tolerance:
        print(f' -> Evaluating RMSE difference with a tolerance of {error_tolerance} days ...')
        if rf_metrics['rmse'] < xgb_metrics['rmse']:
            winner, winning_reason = 'Random Forest', f'Superior RMSE (Margin: {rmse_diff:.3f} days)'
        else:
            winner, winning_reason = 'XGBoost', f'Superior RMSE (Margin: {rmse_diff:.3f} days)'

    else:
        print(f' -> RMSE tied within {error_tolerance} days tolerance. Evaluating MAE difference ...')

        # RANK 2: MAE (The "Translation" Metric)
        mae_diff = abs(rf_metrics['mae'] - xgb_metrics['mae'])

        if mae_diff > error_tolerance:
            if rf_metrics['mae'] < xgb_metrics['mae']:
                winner, winning_reason = 'Random Forest', f'Superior MAE (Margin: {mae_diff:.3f} days)'
            else:
                winner, winning_reason = 'XGBoost', f'Superior MAE (Margin: {mae_diff:.3f} days)'

        else:
            print(f' -> MAE tied within {error_tolerance} days tolerance. Evaluating Computational Time difference ...')

            # RANK 3: TIME (The "Engineering" Tie-Breaker)
            # If accuracy is identical, the fastest model wins to save cloud computing costs.
            if rf_metrics['training_time'] < xgb_metrics['training_time']:
                winner, winning_reason = 'Random Forest', 'Superior Training Time (Tie-breaker)'
            else:
                winner, winning_reason = 'XGBoost', 'Superior Training Time (Tie-breaker)'

    print(f'\n -> Champion Model Selection Complete.')
    print(f' ---> Champion model: {winner.upper()}')
    print(f' ---> Decision logic: {winning_reason}')

    print("\n -> Extracting Top 10 Feature Importances ...")

    if winner == 'Random Forest':
        best_model = best_rf
    else:
        best_model = best_xgb

    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    print(importance_df.to_string(index=False))

    return winner, winning_reason, importance_df