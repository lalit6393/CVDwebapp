import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_sample_data():
    """
    Load the cardiovascular disease dataset. If the file doesn't exist,
    return a small sample dataset with key features.
    """
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

def preprocess_data(data, is_training=False, target_col='cardio'):
    """
    Preprocess the data for model training or prediction.
    
    Parameters:
    data (DataFrame): The input data
    is_training (bool): Whether preprocessing for training
    target_col (str): The name of the target column
    
    Returns:
    X (DataFrame): Preprocessed features
    y (Series, optional): Target variable (only if is_training=True)
    """
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

def train_model(data, model_type='ensemble', model_path=None):
    """
    Train a machine learning model on the dataset.
    
    Parameters:
    data (DataFrame): The input data
    model_type (str): Type of model to train ('rf', 'gb', 'lr', 'svm', 'nn', or 'ensemble')
    model_path (str, optional): Path to save the trained model. If None, uses {model_type}_model.joblib
    
    Returns:
    model: The trained model
    """
    # Set default path based on model type if not provided
    if model_path is None:
        model_path = f"{model_type}_model.joblib"
    # Preprocess the data
    X, y = preprocess_data(data, is_training=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model based on the specified type
    if model_type == 'rf':
        # Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'gb':
        # Gradient Boosting Classifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    elif model_type == 'lr':
        # Logistic Regression
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif model_type == 'svm':
        # Support Vector Machine
        model = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    elif model_type == 'nn':
        # Neural Network
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    elif model_type == 'ensemble':
        # Voting Ensemble of multiple models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
    else:
        # Default to Random Forest if model_type is not recognized
        print(f"Model type '{model_type}' not recognized. Using Random Forest classifier.")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model Performance ({model_type}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save the model
    joblib.dump(model, model_path)
    
    return model

def load_model(model_type='ensemble', model_path=None):
    """
    Load a trained model from disk.
    
    Parameters:
    model_type (str): Type of model to load ('rf', 'gb', 'lr', 'svm', 'nn', or 'ensemble')
    model_path (str, optional): Path to the model file. If None, uses {model_type}_model.joblib
    
    Returns:
    model: The loaded model
    """
    # Set default path based on model type if not provided
    if model_path is None:
        model_path = f"{model_type}_model.joblib"
    
    # Try to load the model from the specified path
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # Try loading from the default model path as fallback
    if model_type != 'default' and os.path.exists('model.joblib'):
        try:
            print(f"Model {model_path} not found, falling back to model.joblib")
            return joblib.load('model.joblib')
        except Exception as e:
            print(f"Error loading default model: {e}")
    
    # If we get here, no valid model could be loaded
    raise FileNotFoundError(f"No valid model found for type {model_type} at path {model_path}")

def load_all_models():
    """
    Load all available trained models.
    
    Returns:
    models (dict): Dictionary of model type to model
    """
    models = {}
    
    # Try to load each type of model
    model_types = ['rf', 'gb', 'lr', 'svm', 'nn', 'ensemble']
    
    for model_type in model_types:
        model_path = f"{model_type}_model.joblib"
        if os.path.exists(model_path):
            try:
                models[model_type] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
    
    # Also try to load the default model
    if os.path.exists('model.joblib'):
        try:
            models['default'] = joblib.load('model.joblib')
        except Exception as e:
            print(f"Error loading default model: {e}")
    
    return models

def make_prediction(model, X, return_all_probs=False):
    """
    Make predictions using the trained model.
    
    Parameters:
    model: The trained model or dictionary of models
    X (DataFrame): Preprocessed input features
    return_all_probs (bool): Whether to return all class probabilities
    
    Returns:
    predictions: The model predictions
    probabilities: The prediction probabilities
    """
    # Check if model is a dictionary (multiple models)
    if isinstance(model, dict):
        all_predictions = {}
        all_probabilities = {}
        ensemble_prob = None
        
        # Make predictions with each model
        for model_type, model_instance in model.items():
            try:
                preds = model_instance.predict(X)
                probs = model_instance.predict_proba(X)
                
                all_predictions[model_type] = preds
                all_probabilities[model_type] = probs
                
                # Accumulate probabilities for ensemble averaging
                if ensemble_prob is None:
                    ensemble_prob = probs
                else:
                    ensemble_prob += probs
            except Exception as e:
                print(f"Error making prediction with {model_type} model: {e}")
        
        # Average the probabilities and make final prediction
        if ensemble_prob is not None and len(model) > 0:
            ensemble_prob /= len(model)
            ensemble_pred = (ensemble_prob[:, 1] >= 0.5).astype(int)
            
            return ensemble_pred, ensemble_prob
        else:
            # Fallback to the first model's predictions if something went wrong
            model_type = next(iter(all_predictions))
            return all_predictions[model_type], all_probabilities[model_type]
    else:
        # Single model prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities

def ensemble_predict(models, X, weights=None):
    """
    Make predictions using multiple models with optional weighting.
    
    Parameters:
    models (dict): Dictionary of models
    X (DataFrame): Preprocessed input features
    weights (dict, optional): Dictionary of model weights
    
    Returns:
    predictions: The ensemble model predictions
    probabilities: The ensemble model probabilities
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    
    all_probabilities = {}
    
    # Get predictions and probabilities from each model
    for model_type, model in models.items():
        try:
            _, probs = make_prediction(model, X, return_all_probs=True)
            all_probabilities[model_type] = probs
        except Exception as e:
            print(f"Error in ensemble prediction with {model_type} model: {e}")
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = {model_type: 1/len(models) for model_type in models}
    
    # Initialize ensemble probabilities
    ensemble_probs = None
    
    # Combine predictions with weights
    for model_type, probs in all_probabilities.items():
        weight = weights.get(model_type, 0)
        if weight > 0:
            if ensemble_probs is None:
                ensemble_probs = weight * probs
            else:
                ensemble_probs += weight * probs
    
    # Convert probabilities to predictions
    if ensemble_probs is not None:
        predictions = (ensemble_probs[:, 1] >= 0.5).astype(int)
        return predictions, ensemble_probs
    else:
        # Fallback to the first model if no valid predictions
        model_type = next(iter(models))
        predictions, probabilities = make_prediction(models[model_type], X)
        return predictions, probabilities
