import json

# Create Notebook
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# House Price Prediction Using Machine Learning"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.datasets import fetch_california_housing\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.metrics import mean_squared_error, r2_score"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "housing = fetch_california_housing()\n",
                "df = pd.DataFrame(housing.data, columns=housing.feature_names)\n",
                "df['Price'] = housing.target\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(df.shape)\n",
                "df.info()\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(df.isnull().sum())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10,8))\n",
                "sns.heatmap(df.corr(), annot=False)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = df.drop('Price', axis=1)\n",
                "y = df['Price']"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = LinearRegression()\n",
                "model.fit(X_train, y_train)\n",
                "print('Model Trained Successfully')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_pred = model.predict(X_test)\n",
                "print(y_pred[:10])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mse = mean_squared_error(y_test, y_pred)\n",
                "r2 = r2_score(y_test, y_pred)\n",
                "print('Mean Squared Error =', mse)\n",
                "print('R2 Score =', r2)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8,6))\n",
                "plt.scatter(y_test, y_pred)\n",
                "plt.xlabel('Actual Price')\n",
                "plt.ylabel('Predicted Price')\n",
                "plt.title('Actual vs Predicted House Price')\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("HousePricePrediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

# requirements.txt
requirements = """pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
"""

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

# README.md
readme = """# House Price Prediction Using Machine Learning

## Project Overview
This project predicts house prices using Machine Learning.

## Dataset
California Housing Dataset

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Model
Linear Regression

## Run Project

pip install -r requirements.txt

jupyter notebook
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

# Project Report
report = """# HOUSE PRICE PREDICTION USING MACHINE LEARNING

## Abstract
This project predicts house prices using Linear Regression and the California Housing Dataset.

## Introduction
House price prediction helps buyers, sellers, and real estate businesses estimate property values.

## Problem Statement
Predict house prices accurately using historical housing data.

## Objectives
- Analyze housing data
- Train a machine learning model
- Predict house prices

## Dataset Description
California Housing Dataset from Scikit-Learn.

## Methodology
1. Data Collection
2. Data Cleaning
3. Data Analysis
4. Model Training
5. Prediction

## Model Used
Linear Regression

## Results
The model predicts house prices with a reasonable R² score.

## Conclusion
Machine learning can effectively estimate housing prices.

## Future Scope
Use advanced algorithms such as Random Forest and XGBoost.

## References
- Scikit-Learn Documentation
- Python Documentation
"""

with open("ProjectReport.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Project files generated successfully!")