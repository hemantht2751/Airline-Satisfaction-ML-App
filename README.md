# ✈️ Airline Passenger Satisfaction Classifier

## 1. Problem Statement
The goal of this project is to build a machine learning model that predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** based on their flight experience. Factors include flight distance, inflight entertainment, seat comfort, and on-board service.

## 2. Dataset Description
* **Source:** Kaggle (Airline Passenger Satisfaction)
* **Input Features:** 22 columns including Gender, Age, Flight Distance, Class, WiFi Service, etc.
* **Target Variable:** `satisfaction` (Binary Classification)
* **Data Processing:** * Missing values in 'Arrival Delay' were imputed with the mean.
    * Categorical features (e.g., Class, Gender) were Label Encoded.
    * Features were scaled using StandardScaler for distance-based algorithms.

## 3. Model Performance Comparison
We trained 6 different classification models. The performance metrics are as follows:

| Model Name           | Accuracy | Precision | Recall | F1 Score | AUC Score | MCC    |
|----------------------|----------|-----------|--------|----------|-----------|--------|
| Logistic Regression  | 0.8779   | 0.8761    | 0.8388 | 0.8570   | 0.9271    | 0.7511 |
| Decision Tree        | 0.9470   | 0.9379    | 0.9408 | 0.9393   | 0.9463    | 0.8922 |
| KNN                  | 0.9296   | 0.9528    | 0.8824 | 0.9163   | 0.9690    | 0.8576 |
| Naive Bayes          | 0.8657   | 0.8645    | 0.8210 | 0.8422   | 0.9228    | 0.7262 |
| Random Forest        | 0.9619   | 0.9742    | 0.9376 | 0.9555   | 0.9937    | 0.9228 |
| XGBoost              | 0.9631   | 0.9723    | 0.9423 | 0.9571   | 0.9949    | 0.9252 |

## 4. Observations & Conclusion

| Model Name | Observation |
|:-----------|:------------|
| **Logistic Regression** | Performed decently (87.8%) but struggled to capture non-linear relationships compared to tree-based models. |
| **Decision Tree** | Strong performance (94.7%) but slightly prone to overfitting compared to the ensemble methods. |
| **KNN** | Good accuracy (92.9%) and high precision, but slower inference time due to distance calculations. |
| **Naive Bayes** | Lowest performance (86.5%), likely because the "independence" assumption doesn't hold true for correlated features like specific service ratings. |
| **Random Forest** | **Excellent performance (96.2%)**. The ensemble approach successfully reduced variance and handled complex patterns. |
| **XGBoost** | **Best Performing Model (96.3%)**. It achieved the highest Accuracy, AUC, and MCC scores, proving to be the most robust classifier for this dataset. |

**Final Choice:** We selected **XGBoost** (or Random Forest) as the primary model for deployment due to its superior F1 score and AUC.