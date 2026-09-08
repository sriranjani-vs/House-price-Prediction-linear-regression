House Price Prediction using Linear Regression

A machine learning project that predicts house prices using Linear Regression.

Project Overview

This project uses the Kaggle House Prices - Advanced Regression Techniques dataset to build a Linear Regression model for predicting house sale prices.

The project includes data preprocessing, handling missing values, categorical feature encoding, feature scaling, model training, cross-validation, and model evaluation.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Linear Regression
- Machine Learning

Machine Learning Process

1. Load the house price dataset
2. Separate features and target variable
3. Apply log transformation to the target
4. Identify numerical and categorical features
5. Handle missing values
6. Scale numerical features
7. Encode categorical features
8. Train a Linear Regression model
9. Perform 5-fold cross-validation
10. Evaluate the model using RMSE, MAE, and R²
11. Save the trained model

Dataset

Dataset: Kaggle House Prices - Advanced Regression Techniques

The dataset contains information about residential properties and their sale prices.

How to Run

1. Download "train.csv" from the Kaggle House Prices dataset.
2. Place "train.csv" in the same folder as "house_price_prediction_linear.py".
3. Install the required libraries:

pip install -r requirements.txt

4. Run the program:

python house_price_prediction_linear.py

Model Evaluation

The model evaluates its performance using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

Model Output

The trained model is saved as:

house_price_model_linear.pkl

Author

Sriranjani V S
B.Tech – Artificial Intelligence and Machine Learning
Lovely Professional University
