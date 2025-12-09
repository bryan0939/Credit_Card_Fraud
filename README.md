#  Credit Card Fraud Detection — Modeling, Data Leakage Investigation & Evaluation

This project builds, evaluates, and interprets machine learning models for detecting fraudulent credit card transactions.  
During development, the analysis uncovered a **leaky feature** that artificially inflated model performance.  
This README documents the full process—from initial modeling to leakage diagnosis and the creation of a realistic, trustworthy fraud detection pipeline.

---

##  Project Overview

Fraud detection is a classic **imbalanced classification problem** where fraudulent transactions represent only a small fraction of total activity.  
This project aims to:

1. Explore and understand the dataset  
2. Establish a baseline classifier  
3. Train and compare multiple ML models  
4. Identify and diagnose **data leakage**  
5. Rebuild the modeling pipeline using only clean features  
6. Evaluate final model performance using appropriate metrics (ROC-AUC, PR-AUC, precision, recall)

---

##  Dataset

The dataset contains anonymized transaction information with the following columns:

- Distance-related features  
- Purchase behavior metrics  
- Boolean indicators (online order, used chip, used PIN, etc.)  
- **Target:** `fraud` (0 = legitimate, 1 = fraudulent)

During modeling, one feature was found to be **highly predictive in an unrealistic way**:

- `ratio_to_median_purchase_price` → **Leaky feature**

This feature was later removed from the clean model.

---

##  Baseline Model

A simple **Most Frequent Class** classifier was used as the baseline.

| Metric | Fraud Class (1) |
|--------|------------------|
| Precision | 0.00 |
| Recall | 0.00 |
| F1-score | 0.00 |

Because fraud is rare, accuracy appears high (~91%) but is completely uninformative.  
Any meaningful model must outperform this baseline.

---

##  Identifying Data Leakage

Initial models showed **near-perfect performance**, which is a major red flag.

Example initial scores (before cleaning):

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Random Forest | ≈ 1.00 | ≈ 1.00 |
| Gradient Boosting | ≈ 0.999 | ≈ 0.999 |
| Logistic Regression | ≈ 0.98 | ≈ 0.75 |

###  Root Cause

Feature importance analysis revealed:

```
ratio_to_median_purchase_price    0.544
distance_from_home                0.200
online_order                      0.106
...
```

The top feature dominated the model and **almost perfectly separated fraud vs. non-fraud**, which should not happen in real-world fraud detection.

This indicates:

- Either **synthetic rule-based data generation**, or  
- **Feature leakage** through a variable built using future information or aggregated statistics

The feature was removed to restore realism.

---

##  Clean Modeling Pipeline (No Leakage)

After dropping the leaky feature, the dataset reflects the true difficulty of the fraud prediction task.

Three models were trained and evaluated using **5-fold stratified cross-validation**:

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Random Forest | **0.787** | **0.447** |
| Gradient Boosting | 0.786 | 0.429 |
| Logistic Regression | 0.784 | 0.344 |

These values are consistent with industry expectations and represent a reliable foundation for further improvement.

---

##  Final Test-Set Performance (Clean Random Forest)

### Confusion Matrix (Normalized)

| True Class | Predicted 0 | Predicted 1 |
|-----------|--------------|-------------|
| Legit (0) | **0.99** | 0.01 |
| Fraud (1) | **0.70** | 0.30 |

### Key Metrics

| Metric | Legit (0) | Fraud (1) |
|--------|-----------|-----------|
| Precision | 0.936 | **0.729** |
| Recall | **0.989** | 0.297 |
| F1-score | 0.962 | 0.422 |

### ROC Curve  
AUC = **0.7848**

### Precision–Recall Curve  
Average Precision = **0.4412**

### Interpretation  
- The model identifies most legitimate transactions correctly  
- It captures ~30% of fraud cases while maintaining strong precision (72.9%)  
- This trade-off is typical in fraud detection and can be improved through threshold tuning, resampling, or cost-sensitive learning

---

##  Key Lessons Learned

- **High performance is not always good**—it may indicate leakage  
- **Feature importance and distribution plots are critical tools** for diagnosing suspicious patterns  
- Realistic fraud detection requires:
  - Correct handling of class imbalance  
  - Proper train/test splitting  
  - Avoiding leakage in feature engineering  
  - Using appropriate evaluation metrics (PR-AUC, recall, etc.)  
- After cleaning, the model provides trustworthy and reproducible performance

---

##  Future Work

- Threshold tuning / cost-sensitive optimization  
- SMOTE or other resampling techniques  
- Ensemble models or stacking  
- SHAP values for interpretability  
- Deploying the model behind an API

---

##  Repository Structure

```
├── Credit_Card_Fraud.ipynb          # Full notebook with leakage diagnosis
├── card_transdata.csv               # Dataset
└── README.md                        # Project documentation
```

---

##  Author  
Bryan Wee Sheng Yuen 
https://github.com/bryan0939

