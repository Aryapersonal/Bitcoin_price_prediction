# Bitcoin Price Prediction Using Machine Learning

##### Dataset : Bitcoin dataset

##### Language: Python,Jupyter Notebook
#### Libraries:

- **Pandas:** For data manipulation and analysis.
- **pandas**: Provides data structures and data analysis tools for handling and manipulating structured data.
- **numpy**: Supports numerical operations and efficient array computations.
- **matplotlib.pyplot**: Enables creating static, interactive, and animated visualizations in Python.
- **seaborn**: Simplifies creating informative and attractive statistical graphics.
- **sklearn.model_selection**: Provides tools for splitting data and performing cross-validation.
- **sklearn.preprocessing**: Includes methods for feature scaling and preprocessing data.
- **sklearn.impute**: Offers strategies for handling missing data through imputation.
- **sklearn.discriminant_analysis**: Implements Linear Discriminant Analysis for classification.
- **sklearn.metrics**: Contains functions for evaluating model performance, including confusion matrix and accuracy score.
- **sklearn.linear_model**: Includes various linear models for regression and classification tasks.
- **sklearn.neighbors**: Provides the K-Nearest Neighbors algorithm for classification and regression.


## Project Overview

This project focuses on predicting Bitcoin prices using various machine learning models to evaluate their performance and reliability. The primary models used are  including LDA ,Logistic Regression, K-Nearest Neighbors (KNN), and Linear Regression. The goal is to determine which model provides the most accurate and reliable predictions for Bitcoin market prices.

**Table of Contents**

1. [Overview](#overview)
2. [Installation](#installation)
3. [Visualizations and Insights](#visualizations-insights)
4. [Key Insights](#key-insights)
5. [Conclusion](#conclusion)
6. [Author Information](#Author-Information)


## Installation
To run this project,  you will need Python and Jupyter Notebook installed on your system.

## Visualization & Insights:

### 1.Linear Discriminant Analysisi (LDA) Model



![download](https://github.com/user-attachments/assets/9e3105d5-a612-4e4d-a548-63c7300c6931)





**Accuracy:**
- **98.97%** accuracy indicates the model's high performance.
  
- **Confusion Matrix:**
- **True Negatives (TN):** 290
- **False Positives (FP):** 0
- **False Negatives (FN):** 6
- **True Positives (TP):** 288

  **Insights:**
- **High Accuracy:** The model is correct 98.97% of the time.
- **No False Positives:** No instances were incorrectly labeled as positive.
- **Low False Negatives:** Only 6 instances were missed as positive.
- **Model Reliability:** Highly effective and reliable for bitcoin price prediction.

### 2. Logistic Regression Model



![download](https://github.com/user-attachments/assets/3c98f5fb-5051-4e4e-b38b-2d88d705a051)




- **Accuracy:** 99.49%
- **Confusion Matrix:**
  - **True Negatives (TN):** 290
  - **False Positives (FP):** 0
  - **False Negatives (FN):** 3
  - **True Positives (TP):** 291
- **Insights:**
  - Logistic Regression shows high accuracy and fewer false negatives compared to LDA.
  - Both models have zero false positives.
  - Overall, Logistic Regression performs slightly better than LDA.

### 3. K-Nearest Neighbors (KNN)


![download](https://github.com/user-attachments/assets/78a8212c-4deb-4263-889c-7b5f0c1fe077)


- **Best k Value:** 1
- **Accuracy:** 99.49%
- **Confusion Matrix:**
  - **True Negatives (TN):** 289
  - **False Positives (FP):** 1
  - **False Negatives (FN):** 2
  - **True Positives (TP):** 292
    
- **Classification Report Metrics:**
  - **Precision:** 0.99 (Class 0), 1.00 (Class 1)
  - **Recall:** 1.00 (Class 0), 0.99 (Class 1)
  - **F1-Score:** 0.99 for both classes
- **Insights:**
  - KNN with k=1 achieves the highest accuracy and minimal misclassifications.
  - It shows high precision and recall, indicating a strong performance in classifying Bitcoin price trends.
  - The accuracy is stable with k values from 1 to around 10, but decreases with larger k values.

### 4. Linear Regression Model

![download](https://github.com/user-attachments/assets/f24ebf86-b4d3-404e-9129-95ba47568f47)



- **Plot Description:**
  - **Actual Prices (Green):** Range from 0 to 1.5.
  - **Predicted Prices (Blue):** Range from -0.3 to 1.
- **Insights:**
  - The linear regression model performs poorly, with predicted values not closely following the actual prices.
  - The model fails to capture the non-linear relationships in Bitcoin prices, leading to scattered predictions.

 ### 5.Actual vs. Predicted Bitcoin Prices (KNN)

![download](https://github.com/user-attachments/assets/cc98dddf-ee3c-4d23-ab32-14be242bb3d3)



 
1. **Comparison**: Actual and predicted Bitcoin prices are closely aligned, especially after index 500, reflecting accurate predictions by the KNN model.
2. **Trends**: The KNN model captures the overall price trend effectively, with predicted prices closely tracking the actual values.
3. **Deviations**: Minor deviations suggest high accuracy; larger deviations are seen in the initial data, indicating areas for potential improvement.
4. **Performance**: The model demonstrates consistent performance over time, showing reliability in price prediction.

###  6.Visualization of Accuracy vs. k for KNN


![download](https://github.com/user-attachments/assets/5ca2acfe-fda9-43c9-8a66-722656cfe646)



- **Figure**:
  - The plot shows accuracy versus different k values for KNN.
  - The x-axis represents k values (1 to 20), and the y-axis shows accuracy.
  - Circular markers highlight accuracy at each k value.

#### Insights

- **Accuracy vs. k Plot**:
  1. **Optimal k**: Highest accuracy at k=1 (~99.49%).
  2. **Accuracy Trend**: High for small k, drops slightly beyond k=10.
  3. **Overfitting/Underfitting**: Small k may overfit; large k may underfit.
  4. **Stability**: Stable accuracy from k=1 to around 10.


   
## Key Insights

- **Best Model:** KNN with k=1 provides the best performance, showing high accuracy, precision, and recall.
- **Comparison:** Logistic Regression is slightly less accurate than KNN but still performs well. Linear Regression does not fit the data effectively.
- **Model Reliability:** KNN is highly reliable for predicting Bitcoin prices, while Linear Regression is less suitable due to its inability to capture complex data patterns.

## Conclusion

The project demonstrates that KNN is the most effective model for predicting Bitcoin prices, offering superior accuracy and reliability. In contrast, Linear Regression proves inadequate for this task due to its poor fit and inability to handle non-linear relationships.

- **LDA Model:** The model demonstrates high performance with an accuracy of 98.97%. It achieves zero false positives and only six false negatives, indicating strong reliability in predicting Bitcoin prices.
- **KNN Model:** Achieved exceptional accuracy (99.49%) with minimal misclassifications and high precision and recall, making it highly reliable for Bitcoin price prediction.
- **Logistic Regression:** Slightly less accurate (99.49%) compared to KNN but still highly effective, showing better performance with fewer false negatives than Linear Discriminant Analysis (LDA).
- **Linear Regression:** Demonstrated weak performance with significant deviations between actual and predicted prices, suggesting it is not suitable for capturing the non-linear patterns in Bitcoin prices.

Overall, KNN emerged as the best model, providing accurate and reliable predictions, while Linear Regression was less effective due to its inability to capture complex data relationships.

### AUTHOR : ARYA S 
### Ping me :www.linkedin.com/in/arya-dataanalyst
### Thank You for reading!

