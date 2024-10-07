# liver-cancer-recurrence
## Analyzing Liver Cancer Recurrence after Surgery Using Machine Learning Models

This study investigates the use of machine learning models to analyze liver cancer (HCC) recurrence after surgery in patients at NTUH between January 2001 and December 2012 (N = 2314). We explore two main outcomes:

1. **Early Recurrence (Binary Classification):** Recurrence within 2 years post-surgery (only patients followed-up for more than 2 years were included).
2. **Recurrence-Free Time (Time-to-Event Outcome):** Recurrence status and time elapsed before recurrence.

**Time-to-Event Regression Models:**

We employed two models for survival analysis:

* **Cox Proportional Hazards Model (CoxPH):** A widely used model assuming a constant effect of explanatory variables on the hazard rate.
* **Random Survival Forests Model (RSF):** An adaptation of random forests for survival analysis that handles right-censored data.

**Key Findings:**

* **Early Recurrence Prediction:** Random Forest exhibited the best performance for predicting recurrence within 2 years (AUC: 0.689 ± 0.013).
* **Recurrence-Free Time Estimation:** RSF demonstrated superior performance over the Cox model for estimating recurrence-free time. Feature selection further improved model performance. Early model performance was lower due to potentially incomplete tumor removal in some patients.
* **Factors Associated with Recurrence:** The Cox model identified several factors associated with increased risk of recurrence (HR > 1, p-value < 0.05):  
    * Tumor size
    * ALBI grade (liver function)
    * Satellite nodules
    * Cirrhosis
    * Surgical margin
    * Tumor stage
    * Age

slides: https://docs.google.com/presentation/d/1FL32_nfl9ia0vp-YJKspW2qhHeIGWtcDIz1tsaKbTeU/edit?usp=sharing
