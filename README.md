# Loan Default Prediction Using Machine Learning

## Overview

This project uses machine learning to predict whether a borrower is likely to default on a loan based on financial and personal information.

The goal is to explore how machine learning can help identify borrowers who may be at higher risk of default. The project focuses on handling an imbalanced classification dataset, comparing different models, engineering meaningful features, and evaluating model performance using appropriate classification metrics.

## Problem Statement

Loan default prediction is an important problem in financial risk assessment. Incorrectly classifying a borrower as low-risk can lead to financial losses, while incorrectly classifying a reliable borrower as high-risk can result in missed lending opportunities.

This project explores whether machine learning models can distinguish between borrowers who default and those who do not.

## Dataset

- **Source:** [Loan Default Dataset on Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- **Task:** Binary classification
- **Target:** Loan default status
- **Features:** Borrower information, income, loan amount, interest rate, loan term, employment duration, and other available attributes.

The target class represents whether a borrower defaulted on a loan. Refer to the dataset's documentation for the exact target encoding.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook / Google Colab

## Project Workflow

1. **Data exploration:** Examined the dataset, feature distributions, and class imbalance.
2. **Data preprocessing:** Prepared numerical and categorical features using scaling and encoding where appropriate.
3. **Baseline modeling:** Trained and compared Logistic Regression, Random Forest, and XGBoost.
4. **Hyperparameter tuning:** Used cross-validation and randomized search to explore model configurations.
5. **Feature engineering:** Created additional features, including:
   - Loan-to-income ratio
   - Employment duration in years
   - Estimated monthly payment burden relative to monthly income
6. **Imbalanced classification:** Experimented with class weighting and decision-threshold tuning.
7. **Model evaluation:** Compared models using precision, recall, F1-score, ROC-AUC, and average precision (PR-AUC).
8. **Model interpretability:** Explored SHAP to understand how features influence model predictions.

## Feature Engineering

Three additional features were explored:

- **Loan-to-income ratio:** Compares the loan amount with the borrower's income.
- **Employment years:** Converts employment duration from months into years.
- **Estimated monthly payment burden:** Estimates the monthly payment burden relative to estimated monthly income.

These features were designed to represent aspects of a borrower's potential repayment capacity. The monthly payment calculation is an approximation rather than a full loan amortization calculation.

## Model Performance

The best reported result in this experiment came from the feature-engineered XGBoost model with a tuned decision threshold.

| Metric | Score |
|---|---:|
| Accuracy | 0.8065 |
| Precision | 0.2950 |
| Recall | 0.4854 |
| F1-score | 0.3670 |
| ROC-AUC | 0.7548 |
| PR-AUC (Average Precision) | 0.3264 |

*These results are from the current experiment and should be verified against the final evaluation code before publication.*

### What do the results mean?

- **Precision:** Approximately 29.5% of borrowers predicted as defaulters were actual defaulters.
- **Recall:** The model identified approximately 48.5% of actual defaulters.
- **F1-score:** Balances precision and recall, helping assess performance when the classes are imbalanced.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across thresholds.
- **PR-AUC:** Summarizes precision-recall performance and is particularly useful when the positive class is relatively uncommon.

The results demonstrate that predicting loan defaults remains challenging. The model's decision threshold affects the balance between identifying more defaulters and incorrectly flagging borrowers who would not default.

## Key Learnings

Through this project, I gained practical experience with:

- Building end-to-end machine learning pipelines
- Preprocessing mixed numerical and categorical data
- Handling imbalanced classification problems
- Comparing Logistic Regression, Random Forest, and XGBoost
- Performing cross-validation and hyperparameter tuning
- Engineering features based on domain knowledge
- Tuning decision thresholds
- Evaluating models with precision, recall, F1-score, ROC-AUC, and PR-AUC
- Exploring model interpretability using SHAP

## Limitations

- The model does not perfectly identify borrowers who will default.
- Its performance depends on the dataset's quality and representativeness.
- Feature-engineered payment estimates may not reflect actual loan repayment schedules.
- The model has not been established as suitable for real-world lending decisions.
- Further validation, calibration, fairness analysis, and testing on unseen data would be needed before any real-world use.

## Future Improvements

- Explore additional feature engineering.
- Compare alternative models and tuning strategies.
- Improve interpretability and probability calibration.
- Evaluate performance across relevant borrower groups.
- Deploy the complete preprocessing and prediction pipeline through an API.

## Conclusion

This project provided hands-on experience with the practical challenges of building a machine learning model for an imbalanced financial classification problem. Rather than relying on accuracy alone, it evaluates the trade-offs between precision and recall and explores how feature engineering and decision-threshold tuning affect model performance.

The project is an educational machine learning experiment and not a production-ready credit risk system.
