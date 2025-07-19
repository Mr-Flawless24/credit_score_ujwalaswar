#
# Disease Prediction from Medical Data using UCI Heart Disease Dataset
#
# This script builds and evaluates four different machine learning models
# (Logistic Regression, SVM, Random Forest, XGBoost) to predict the
# presence of heart disease based on patient medical data.
#
# The process includes:
# 1. Loading the Heart Disease dataset from the UCI repository.
# 2. Cleaning and preprocessing the data (handling missing values).
# 3. Defining preprocessing steps for numerical and categorical features.
# 4. Creating and training machine learning pipelines for each model.
# 5. Evaluating the models using key classification metrics.
#

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def build_and_evaluate_disease_models():
    """
    Main function to load data, build, train, and evaluate disease prediction models.
    """
    # --- 2. Load and Prepare the Dataset ---
    print("Step 1: Loading and Preparing Data...")
    
    # The dataset is hosted at the UCI Machine Learning Repository.
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    
    # Define the column names as per the dataset's documentation.
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load the data using pandas. Missing values are denoted by '?'.
    try:
        data = pd.read_csv(url, header=None, names=column_names, na_values='?')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 3. Data Cleaning and Preprocessing ---
    print("\nStep 2: Cleaning and Preprocessing Data...")

    # The target variable indicates the presence of heart disease.
    # Values > 0 mean disease is present. We'll map it to a binary format: 0 (no disease) vs 1 (disease).
    data['target'] = (data['target'] > 0).astype(int)

    # Separate features (X) and the target variable (y).
    X = data.drop('target', axis=1)
    y = data['target']

    # Identify numerical and categorical features.
    # Note: Some features are numerically coded but are actually categorical.
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    print(f"Identified {len(numerical_features)} numerical features: {numerical_features}")
    print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

    # Create preprocessing pipelines for both data types.
    # For numerical data, we impute missing values with the median and then scale.
    # For categorical data, we impute with the most frequent value and then one-hot encode.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer to apply different transformations.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 4. Split Data ---
    print("\nStep 3: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # --- 5. Define and Train Models ---
    print("\nStep 4: Defining and Training Models...")
    
    # We define each model within a Pipeline.
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ]),
        "Support Vector Machine": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42)) # probability=True for ROC-AUC
        ]),
        "Random Forest": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "XGBoost": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    }

    # Train each model.
    for name, pipe in pipelines.items():
        print(f"  - Training {name}...")
        pipe.fit(X_train, y_train)
    print("All models trained successfully!")

    # --- 6. Evaluate Models ---
    print("\nStep 5: Evaluating Model Performance on Test Data...")

    for name, model in pipelines.items():
        print(f"\n{'='*30}")
        print(f"RESULTS FOR: {name}")
        print(f"{'='*30}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate and print metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)']))

if __name__ == '__main__':
    build_and_evaluate_disease_models()
