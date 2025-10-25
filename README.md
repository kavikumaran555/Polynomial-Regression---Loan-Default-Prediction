# Polynomial-Regression---Loan-Default-Prediction
Polynomial Regression - Loan Default Prediction
Overview

This project uses polynomial regression to predict the probability of loan default for customers based on three key features: credit score, annual income, and loan amount. The model captures nonlinear relationships between input features and the target variable, allowing better prediction for complex patterns.

Dataset

The dataset contains 50 rows with the following columns:

Credit_Score: Customer's credit score

Annual_Income: Customer's yearly income

Loan_Amount: Loan requested by the customer

Loan_Default_Probability: Target variable, probability of loan default

Libraries Used

pandas for data handling

numpy for numerical operations

matplotlib.pyplot for visualization

sklearn.linear_model.LinearRegression for building the regression model

sklearn.preprocessing.PolynomialFeatures for polynomial transformation

How the Code Works

Load the dataset.

Assign independent variables (x) and dependent variable (y).

Transform features using polynomial features (degree 2).

Fit the Linear Regression model.

Predict the loan default probabilities.

Visualize actual vs predicted values on a scatter plot.

Calculate R² to measure the model’s fit.

Example Plot

The plot shows blue dots for actual default probabilities and red dots for predicted probabilities.

R² Score

The model’s R² indicates how well the polynomial regression explains the variation in the target variable. A value close to 1 indicates a strong fit.
