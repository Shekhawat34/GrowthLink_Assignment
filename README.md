# Titanic Survival Prediction

This project builds a machine learning model to predict passenger survival from the Titanic disaster using passenger data such as age, gender, ticket class, and fare.

# Table of Content
1. Project Overview
2. Dataset Description
3. Installation
4. Project Structure
5. Data Preprocessing
6. Feature Engineering
7. Model Selection
8. Model Evaluation
9. Results
10. Usage

# Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project aims to build a predictive model that determines which passengers survived the disaster based on passenger attributes. This is implemented as a binary classification problem.


# Dataset Description
The Dataset Includes the following Fields:

1. PassengerId: Unique identifier for each passenger
2. Survived: target variable (0=No,1=Yes)
3. PClass: Ticket Class (1=1st class, 2=2nd class, 3=3rd class)
4. Name: Passenger name
5. Sex: Passenger gender
6. Age: Passenger age
7. SibSp: Number of siblings/spouses aboard
8. Parch: number of parents/children aboard
9. Ticket: Ticket number
10. Fare: Passenger Fare


# Installation

# Clone the repository
git clone https://github.com/Shekhawat34/GrowthLink_Assignment/titanic-survival-prediction.git
cd titanic-survival-prediction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Project Structure

titanic-survival-prediction/
│
├── data/                   # Data directory
│   ├── train.csv           # Training data
│             
│
├── models/                 # Saved model files
│   └── titanic_model.pkl   # Trained model
│
├── notebooks/              # Jupyter notebooks
│   └── Titanic_Survival_Prediction.ipynb           # Exploratory Data Analysis
│
├── images/                 # Visualizations
│   ├── missing_values.png
│   ├── survival_analysis.png
│   ├── confusion_matrix.png
│   └── feature_importances.png
│
|     
└── README.md               # Project documentation


# Data Preprocessing

The data preprocessing pipeline includes:

1. handling the missing values:
2. Categorical Encoding
3. Numerical Scaling
4. Data Spliting (75% training, 25% testing)


# Feature Engineering
Several features were engineered to improve model performance:

1. TItle Extraction
2. Family Features (Sum of SibSp+Parch+1(Self))
3. Cabin Information


# Model Selection

We employed a Random Forest Classifier with hyperparameter tuning through GridSearchCV to find the optimal model configuration. The parameters tuned include:

1. Number of estimators (trees): [100,200]
2. Maximum depth: [None,10,15]
3. Minimum samples for split :[2,5]
   

The entire preprocessing and modeling pipeline was built using scikit-learn's Pipeline and ColumnTransformer to ensure proper handling of different feature types and prevent data leakage.


# Model Evaluation

1. Acuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

# Usage

import pickle

#Load the model
with open('models/titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Prepare data (ensure it's in the same format as training data)
#...

# Make predictions
predictions = model.predict(new_data)





