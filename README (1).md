### Diabetes-Patient-Readmission-Classification
develop a pattern recognition system that operates on a given real- world dataset. 

### Introduction
The goal of this project is to develop a pattern recognition system that operates on a given real-
world dataset.

### Dataset
Our dataset is: Diabetes 130-US hospitals for years 1999-2008 Data Set.

Source:
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-800

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated
delivery networks. It includes over 50 features representing patient and hospital outcomes. 

The target labels, indicating: 
* “<30” if the patient was readmitted in less than 30 days 
* “>30” if the patient was readmitted in more than 30 days 
* “No” for no record of readmission.

This problem is a 3-class classification problem.
![image](/Users/apple/Desktop/Projects/Final_Project_ML_Akshay_Mahto/diabetes_SourceCode/diabetes_drugs.png)
![image](/Users/apple/Desktop/Projects/Final_Project_ML_Akshay_Mahto/diabetes_SourceCode/diabetes_TimeInHospital.png)

### Project Goal
The main goal of this project is to design a machine learning classification system, that is able to predict the readmission of a diabetes patient, based on the patient's medical history information.

### Purpose and Key Functionality

The aim of this project is to analyze, over a period of ten years, the hospital records of diabetic patients to identify trends that lead to readmissions. Main functionalities include a predictive model in classifying readmissions by "<30 days," ">30 days," and "No readmission." This project leveraged data obtained from the UCI Machine Learning Repository: the Diabetes 130-US hospitals dataset containing patient demographic information, such as medical history, laboratory test results, and medication dosage, among other factors. Therefore, in building this model, the objective is to enhance decision making with major cleaning and preprocessing of the dataset, applying exploratory analyses, and building classification models of the patients.

### Setup and Execution

Prerequisites please check with below which is requirement for this project

(i)Python 3.10

(ii)Libraries:

(iii)numpy

(iii)pandas

(iv)matplotlib

(v)seaborn

(vi)scikit-learn

Ensure the dataset diabetic_data.csv is in the working directory.

### Installation

Clone the repository:

git clone <repository_url>

Install required libraries:

pip install -r requirements.txt

Execution Steps

Load and preprocess the dataset using the provided data cleaning scripts.

Split the data into training, validation, and test sets.

Train models:

Naive Bayes

k-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest

Gradient Boosting

Evaluate the models on the validation set and tune hyperparameters (GridSearchCV is used).

Test the final model on the test set and output the results.

To run the main script:

python main.py

Reproducing Results

### Data Preprocessing:

Follow the steps in the data cleaning script for handling missing values and encoding categorical variables.

Scale the data by using StandardScaler from scikit-learn.

### Model Training:

Run the scripts for training of individual models and tuning their hyperparameters.

Use GridSearchCV for tuning Gradient Boosting hyperparameters, such as n_estimators, max_depth, and learning_rate.

### Evaluation:

Use classification_report and confusion matrices for evaluating the predictions.

Performance metrics: F1 Score and accuracy. Visualizations: Generate plots for feature importance and data correlations using matplotlib and seaborn.



### Conclusion
We have acheived the best prediction performance using Gradient Boost classifier.
* F1 Score (micro): 0.6215
* F1 Score (macro): 0.3612

The main reasons for not acheiving a high classification performance is the fact that our labels are not palnced thoughout
the dataset, where 1 label (No readmission) accounts for over 60% of the data points, while another
label (Readmitted in < 30 days) accounts for only ~8%. Another reason for low performance is
that our target has very low correlation with all of our predectors.
![image](/Users/apple/Desktop/Projects/Final_Project_ML_Akshay_Mahto/diabetes_SourceCode/diabetes_Results.png)

### Future Work 
In the future, we can try the following to improve the performance of our classifier:
1. combine (readmitted in < 30 days) and (readmitted in > 30 days) into one feature, and turn
the problem in to binary classification problem (readmitted vs not readmmited) which is going to
result in a better palanced label classes. 
2. We can try selecting a subset of feature, with higher
classification importance according to our classifier, and just using those in training.

### References
* https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
* https://www.hindawi.com/journals/bmri/2014/781670/
* https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python