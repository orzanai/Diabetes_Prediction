
# Diabetes Prediction

The main goal of this project is to develop a machine-learning model that can predict whether individuals have diabetes or not when their features are specified.
Before developing the model, the necessary data analysis and feature engineering steps are performed.
## Dataset

The dataset is a part of a large dataset maintained at the National Institute of Diabetes and Digestive and Kidney Diseases in the United States. It is used for a diabetes study conducted on Pima Indian women aged 21 and above living in Phoenix, the fifth largest city in the state of Arizona in the US. The target variable is specified as "outcome", where 1 indicates a positive diabetes test result and 0 indicates a negative result.




## Methods

- Exploratory Data Analysis
- Outlier Handling with IQR and LOF(Local Outlier Factor)
- Missing Values Imputation with KNN and Mean/Median
- Feature Extraction
- Encoding with Label and One-Hot Encoder
- Feature Scaling
- Random Forest Classifier Model





## Results

At the beginning of the study, a Random Forest Classifier model was built, and an accuracy score of 0.77 was obtained. However, after the feature engineering process was performed and a new model was rebuilt at the end of the study, the accuracy score was calculated as 0.88. The most effective result in filling in missing values was achieved by using the median value. As for the extracted variables, the categorical ones were found to be ordinal; therefore, label encoding was used, and the model with label encoding output outperformed the one with one-hot encoding. The least affected scaling method by outliers, robust scaler, was used. The variables obtained from feature extraction were observed to be among the top in importance order.


## Acknowledgements

This project was completed as a part of the Miuul Data Science & Machine Learning Bootcamp.
