#
# Credit Scoring Model using German Credit Data
#
# This script builds and evaluates three different machine learning models
# (Logistic Regression, Decision Tree, and Random Forest) to predict
# creditworthiness based on a publicly available dataset.
#
# The process includes:
# 1. Loading and preparing the data.
# 2. Defining preprocessing steps for numerical and categorical features.
# 3. Creating and training machine learning pipelines.
# 4. Evaluating the models using key classification metrics.
#

# --- 1. Import Necessary Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

def build_and_evaluate_credit_models():
    """
    Main function to load data, build, train, and evaluate credit scoring models.
    """
    # --- 2. Load and Prepare the Dataset ---
    print("Step 1: Loading and Preparing Data...")
    
    # The dataset is hosted at the UCI Machine Learning Repository.
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    
    # Define the column names as per the dataset's documentation.
    column_names = [
        'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'present_employment', 'installment_rate', 'personal_status',
        'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk'
    ]
    
    # Load the data using pandas. It's a space-separated file.
    try:
        data = pd.read_csv(url, sep=' ', header=None, names=column_names)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # The original target 'credit_risk' is 1 for good, 2 for bad.
    # We'll convert it to the standard 0 (good) and 1 (bad) for easier interpretation.
    data['credit_risk'] = data['credit_risk'].map({1: 0, 2: 1})
    print("Target variable 'credit_risk' mapped to 0 (Good) and 1 (Bad).")

    # Separate features (X) and the target variable (y).
    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk']

    # --- 3. Preprocessing ---
    print("\nStep 2: Setting up Preprocessing...")

    # Identify numerical and categorical features for separate processing.
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print(f"Identified {len(numerical_features)} numerical features: {list(numerical_features)}")
    print(f"Identified {len(categorical_features)} categorical features: {list(categorical_features)}")

    # Create a preprocessing pipeline using ColumnTransformer.
    # - Numerical features will be scaled.
    # - Categorical features will be one-hot encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # --- 4. Split Data ---
    print("\nStep 3: Splitting data into training and testing sets...")
    # We split the data into 80% for training and 20% for testing.
    # 'stratify=y' ensures the proportion of good/bad credit is the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # --- 5. Define and Train Models ---
    print("\nStep 4: Defining and Training Models...")
    
    # We define each model within a Pipeline to chain preprocessing and classification.
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ]),
        "Decision Tree": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    # Train each model in the dictionary.
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
        
        # Make predictions on the test set.
        y_pred = model.predict(X_test)
        
        # For ROC-AUC, we need prediction probabilities for the positive class (1).
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        except AttributeError:
            # Some models might not have predict_proba
            print("ROC-AUC Score: Not available for this model.")

        # Calculate other key metrics.
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}\n")
        
        # Print a detailed classification report.
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Good Credit (0)', 'Bad Credit (1)']))

if __name__ == '__main__':
    build_and_evaluate_credit_models()
