import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_sample_data():
    try:
        # Try to load the dataset
        data = pd.read_csv('dataset/cardio_dataset.csv', delimiter=';')
        return data
    except:
        # Create a small sample dataset for demonstration
        print("Creating sample dataset...")
        # This is a synthetic sample dataset for illustration only
        np.random.seed(42)
        n_samples = 500
        
        # Generate age in days (from 40 to 70 years)
        age = np.random.randint(40*365, 70*365, n_samples)
        
        # Generate gender (1 for female, 2 for male)
        gender = np.random.choice([1, 2], n_samples)
        
        # Generate height (cm)
        height = np.random.normal(165, 10, n_samples).astype(int)
        
        # Generate weight (kg)
        weight = np.random.normal(75, 15, n_samples).astype(int)
        
        # Generate blood pressure
        ap_hi = np.random.normal(120, 15, n_samples).astype(int)  # Systolic
        ap_lo = np.random.normal(80, 10, n_samples).astype(int)   # Diastolic
        
        # Generate cholesterol (1:normal, 2:above normal, 3:well above normal)
        cholesterol = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1])
        
        # Generate glucose (1:normal, 2:above normal, 3:well above normal)
        gluc = np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1])
        
        # Generate binary features
        smoke = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        alco = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        active = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        
        # Generate target variable (CVD presence)
        # Higher probability of CVD with increased age, BP, cholesterol
        cvd_prob = 0.3 + 0.3 * (age > 55*365) + 0.2 * (ap_hi > 140) + 0.2 * (cholesterol > 1)
        cvd_prob = np.clip(cvd_prob, 0, 0.9)  # Clip probabilities
        cardio = np.random.binomial(1, cvd_prob)
        
        # Create DataFrame
        data = pd.DataFrame({
            'id': range(1, n_samples+1),
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'cardio': cardio
        })
        
        # Create directory if it doesn't exist
        os.makedirs('dataset', exist_ok=True)
        
        # Save the dataset
        data.to_csv('dataset/cardio_dataset.csv', index=False, sep=';')
        
        return data

def preprocess_data(data, is_training=True, target_col='cardio'):
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop ID column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Handle outliers for numerical columns
    # Blood pressure outliers
    df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 160)]
    
    # Height and weight outliers
    df = df[(df['height'] >= 120) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    
    # Feature engineering
    # BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Blood pressure categories
    df['bp_category'] = 0  # Normal
    df.loc[(df['ap_hi'] >= 140) | (df['ap_lo'] >= 90), 'bp_category'] = 1  # Hypertension
    
    # Age in years
    df['age_years'] = df['age'] / 365
    
    # Pulse pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Separate features and target
    if is_training and target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df
        y = None
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_years', 'pulse_pressure']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    if is_training:
        return X, y
    else:
        return X

def train_xgb_model():
    print("Loading sample data...")
    data = load_sample_data()
    
    print("Preprocessing data...")
    X, y = preprocess_data(data, is_training=True)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"XGBoost Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("Saving model...")
    joblib.dump(model, 'xgb_model.joblib')
    print("XGBoost model saved successfully!")
    
    return model

if __name__ == "__main__":
    train_xgb_model()