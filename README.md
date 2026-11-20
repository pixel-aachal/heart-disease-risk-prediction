# Heart Disease Risk Prediction

Machine learning project that predicts heart disease risk from 13 clinical
features (Statlog heart disease dataset, 270 patients).

**Live demo:** https://huggingface.co/spaces/aachalgupta09/heart-disease-risk

## What I did
- Data cleaning and EDA (distributions, correlation heatmap)
- Trained Decision Tree, Random Forest and Logistic Regression
- Evaluated with accuracy, recall, ROC-AUC, confusion matrix and 5-fold cross-validation
- Exported the final model's coefficients to JSON and deployed a
  browser-based predictor on Hugging Face Spaces

## Results
| Model | 5-fold CV ROC-AUC |
|---|---|
| Logistic Regression | 0.896 ± 0.044 |
| Random Forest | 0.877 ± 0.049 |
| Decision Tree | 0.719 ± 0.063 |

Final model: Logistic Regression. On the held-out test set (54 patients):
accuracy 0.85, ROC-AUC 0.90, disease recall 0.92 (22 of 24 caught).

## Files
- `Heart_Disease_Risk_Prediction.ipynb`: analysis and training
- `heart_disease_model.pkl`: trained model, scaler and feature list
- `index.html`, `model.json`: web demo (runs in the browser)

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn, HTML/JavaScript

## Limitations
Small dataset (270 rows) and small test set, so scores are noisy.
Educational project only, not a medical tool.
