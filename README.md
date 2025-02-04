# Student Loan Repayment Prediction using Neural Networks

## Overview
This project develops a deep learning model to predict whether a student loan applicant is likely to repay their loan. The model uses historical data and features related to student loan recipients to predict credit ranking. By accurately predicting repayment likelihood, the company can offer better interest rates to borrowers.

## Dataset
The dataset contains information about student loan recipients, including various financial indicators and a `credit_ranking` column, which serves as the target variable.

### Features Used:
- Multiple financial and demographic features
- `credit_ranking` as the target variable

## Model Development
### **Step 1: Data Preprocessing**
- Loaded the dataset into a Pandas DataFrame.
- Split the dataset into features (`X`) and target (`y`).
- Used `StandardScaler` from scikit-learn to normalize the feature set.
- Split the data into training and testing sets.

### **Step 2: Neural Network Model**
- Designed a deep neural network using TensorFlow and Keras.
- Used two hidden layers with the ReLU activation function.
- Compiled the model using `binary_crossentropy` loss function, `adam` optimizer, and `accuracy` as the evaluation metric.
- Trained the model for 50 epochs.

### **Step 3: Model Evaluation**
- Evaluated the model using test data.
- Achieved a model accuracy of **0.7500** (75%).
- Achieved a model loss of **0.5508**.

### **Step 4: Predictions and Classification Report**
- Reloaded the saved model (`student_loans.keras`).
- Used the model to predict loan repayment likelihood.
- Rounded predictions to binary values (0 or 1).
- Generated a classification report for performance analysis.

### **Model Performance Metrics**
```
Classification Report: 
              precision    recall  f1-score   support

           0       0.73      0.77      0.75       154
           1       0.78      0.73      0.75       166

    accuracy                           0.75       320
   macro avg       0.75      0.75      0.75       320
weighted avg       0.75      0.75      0.75       320
```

### **Interpretation of Results**
- **Precision**: The model correctly predicts class **0 (non-repayment)** with 73% precision and class **1 (repayment)** with 78% precision.
- **Recall**: The model successfully identifies 77% of actual non-repayers and 73% of actual repayers.
- **F1-Score**: A balanced measure of precision and recall, both at 0.75, indicating a well-performing model.
- **Overall Accuracy**: The model correctly classifies **75%** of cases, making it a reliable tool for student loan assessment.

## Recommendations for a Student Loan Recommendation System

### **Data Collection for Loan Recommendation**
To create a recommendation system for student loans, relevant data should include:
- **Credit score**: Determines loan eligibility and interest rates.
- **Income level**: Helps assess repayment ability.
- **Employment status**: Indicates financial stability.
- **Debt-to-income ratio**: Evaluates financial risk.
- **Loan amount requested**: Customizes loan offers.
- **Repayment history**: Influences interest rates and loan approval.

These factors allow the system to suggest personalized loan options based on the applicant’s financial profile.

### **Filtering Method Selection**
A **content-based filtering** approach is suitable since loan recommendations depend on user-specific financial attributes rather than collaborative user behaviors.

### **Challenges in Building a Loan Recommendation System**
1. **Data Privacy & Security**: Financial data must be securely handled to comply with regulations (e.g., GDPR, CCPA).
2. **Bias in Data**: Historical biases in loan approvals may result in unfair recommendations, requiring careful bias mitigation strategies.

## Conclusion
This project successfully builds a neural network to predict student loan repayment likelihood. The model achieves 75% accuracy and demonstrates a strong balance between precision and recall. Future enhancements could involve integrating additional financial features and testing alternative machine learning models to improve predictive performance.

---
## How to Run the Project
1. Install required dependencies: `pip install pandas scikit-learn tensorflow`
2. Run the Jupyter Notebook to preprocess data, train the model, and evaluate performance.
3. Load the saved model and generate predictions.

## Files in Repository
- `student_loans.ipynb` - Jupyter Notebook containing data processing, model training, and evaluation.
- `student_loans.keras` - Saved trained model for future use.

