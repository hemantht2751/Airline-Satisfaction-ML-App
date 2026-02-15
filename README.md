# ✈️ Airline Passenger Satisfaction Classifier

## 1. Problem Statement
The goal of this project is to build a machine learning model that predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** based on their flight experience. Factors include flight distance, inflight entertainment, seat comfort, and on-board service.

## 2. Dataset Description
* **Source:** Kaggle (Airline Passenger Satisfaction)
* **Input Features:** 22 columns (Satisfies assignment requirement of >12 features). Features include Gender, Age, Flight Distance, Class, WiFi Service, etc.
* **Target Variable:** `satisfaction` (Binary Classification)
* **Data Processing:** * Missing values in 'Arrival Delay' were imputed with the mean.
    * Categorical features (e.g., Class, Gender) were Label Encoded.
    * Features were scaled using StandardScaler for distance-based algorithms (KNN, Logistic Regression).

## 3. Model Performance Comparison
We trained 6 different classification models. The performance metrics are sorted as per the assignment requirements:

| Model Name           | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC    |
|----------------------|----------|-----------|-----------|--------|----------|--------|
| Logistic Regression  | 0.8779   | 0.9271    | 0.8761    | 0.8388 | 0.8570   | 0.7511 |
| Decision Tree        | 0.9470   | 0.9463    | 0.9379    | 0.9408 | 0.9393   | 0.8922 |
| KNN                  | 0.9296   | 0.9690    | 0.9528    | 0.8824 | 0.9163   | 0.8576 |
| Naive Bayes          | 0.8657   | 0.9228    | 0.8645    | 0.8210 | 0.8422   | 0.7262 |
| Random Forest        | 0.9619   | 0.9937    | 0.9742    | 0.9376 | 0.9555   | 0.9228 |
| XGBoost              | 0.9631   | 0.9949    | 0.9723    | 0.9423 | 0.9571   | 0.9252 |

## 4. Observations & Conclusion

| Model Name | Observation |
|:-----------|:------------|
| **Logistic Regression** | This model provided a solid baseline with 87.8% accuracy but struggled to capture complex, non-linear relationships between service ratings and satisfaction. It had a lower MCC (0.75) compared to tree-based models, indicating it is less robust in distinguishing classes perfectly. |
| **Decision Tree** | Achieved strong performance (94.7%) by capturing non-linear patterns, but the Recall (94.08%) was slightly lower than XGBoost. Single decision trees are prone to overfitting, which explains why its AUC (0.94) is lower than the ensemble methods. |
| **KNN** | KNN delivered good accuracy (92.9%) and a very high AUC (0.969). However, it is computationally expensive during inference as it calculates distances for every prediction. It performed well here because the features were scaled correctly. |
| **Naive Bayes** | This model had the lowest performance (86.5% Accuracy, 0.72 MCC). This is likely because the "feature independence" assumption is violated; service ratings (e.g., Food vs. Cleanliness) are often correlated, which confuses the model. |
| **Random Forest** | **Excellent performance (96.2%)**. By aggregating multiple trees (Bagging), it successfully reduced the variance seen in the single Decision Tree. It achieved a near-perfect AUC of 0.9937, making it a highly reliable candidate for deployment. |
| **XGBoost** | **Best Performing Model (96.3%)**. Using Gradient Boosting, it iteratively corrected errors from previous trees to achieve the highest Precision (97.2%) and MCC (0.925). It is the most robust model for this dataset, handling both bias and variance effectively. |

**Final Choice:** We selected **XGBoost** as the primary model for deployment due to its superior F1 score and AUC, ensuring the most accurate passenger satisfaction predictions.
