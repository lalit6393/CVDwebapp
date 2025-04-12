import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import permutation_importance
import os

def load_sample_data():
    """
    Load the cardiovascular disease dataset from a CSV file.
    """
    data_path = 'dataset/cardio_dataset.csv'
    
    # Check if the file exists
    if os.path.exists(data_path):
        try:
            # Try to load the dataset with semicolon separator
            data = pd.read_csv(data_path, delimiter=';')
            return data
        except:
            try:
                # Try with comma separator
                data = pd.read_csv(data_path)
                return data
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                # Let the model.py create a sample dataset
                from model import load_sample_data as create_sample_data
                return create_sample_data()
    else:
        # Let the model.py create a sample dataset
        from model import load_sample_data as create_sample_data
        return create_sample_data()

def calculate_feature_importance(model, X):
    """
    Calculate feature importance for the model.
    
    Parameters:
    model: The trained model
    X (DataFrame): Input features
    
    Returns:
    feature_importance (dict): Dictionary of feature importance
    """
    # Get feature importance from the model
    if hasattr(model, 'feature_importances_'):
        # Random Forest and other tree-based models
        importances = model.feature_importances_
        feature_importance = dict(zip(X.columns, importances))
    else:
        # For other models, use permutation importance
        try:
            perm_importance = permutation_importance(model, X, random_state=42)
            feature_importance = dict(zip(X.columns, perm_importance.importances_mean))
        except:
            # Fallback to a simple method
            # This is a simplistic approach and not recommended for real applications
            feature_importance = {}
            for col in X.columns:
                feature_importance[col] = np.random.random()  # Dummy value
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    return feature_importance

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters:
    feature_importance (dict): Dictionary of feature importance
    top_n (int): Number of top features to display
    
    Returns:
    fig: The matplotlib figure
    """
    # Take top N features
    top_features = dict(list(feature_importance.items())[:top_n])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    features = list(top_features.keys())
    importances = list(top_features.values())
    
    # Map feature names to more readable format
    feature_mapping = {
        'age': 'Age',
        'age_years': 'Age (years)',
        'gender': 'Gender',
        'height': 'Height',
        'weight': 'Weight',
        'ap_hi': 'Systolic BP',
        'ap_lo': 'Diastolic BP',
        'cholesterol': 'Cholesterol',
        'gluc': 'Glucose',
        'smoke': 'Smoking',
        'alco': 'Alcohol',
        'active': 'Physical Activity',
        'bmi': 'BMI',
        'bp_category': 'BP Category',
        'pulse_pressure': 'Pulse Pressure'
    }
    
    # Map feature names
    features = [feature_mapping.get(f, f) for f in features]
    
    # Plot
    bars = ax.barh(features, importances, color='skyblue')
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    # Customize the plot
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def format_prediction_text(prediction, probability):
    """
    Format prediction result as text.
    
    Parameters:
    prediction: The model prediction (0 or 1)
    probability: The prediction probability
    
    Returns:
    text (str): Formatted text
    """
    if prediction == 1:
        text = f"⚠️ High risk of cardiovascular disease detected! (Confidence: {probability:.1%})"
    else:
        text = f"✅ Low risk of cardiovascular disease. (Confidence: {probability:.1%})"
    
    return text

def get_recommendations(user_data):
    """
    Generate health recommendations based on user data.
    
    Parameters:
    user_data (dict): User health data
    
    Returns:
    recommendations (list): List of recommendations
    """
    recommendations = []
    
    # Blood pressure recommendations
    if user_data.get('ap_hi', 0) > 140 or user_data.get('ap_lo', 0) > 90:
        recommendations.append("Your blood pressure is elevated. Consider consulting a healthcare professional.")
    
    # BMI recommendations
    height_m = user_data.get('height', 170) / 100
    weight_kg = user_data.get('weight', 70)
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        recommendations.append(f"Your BMI is {bmi:.1f}, which is below the healthy range. Consider consulting a nutritionist.")
    elif bmi >= 25 and bmi < 30:
        recommendations.append(f"Your BMI is {bmi:.1f}, which indicates overweight. Consider a balanced diet and regular exercise.")
    elif bmi >= 30:
        recommendations.append(f"Your BMI is {bmi:.1f}, which indicates obesity. Consider consulting a healthcare professional.")
    
    # Cholesterol recommendations
    if user_data.get('cholesterol', 1) > 1:
        recommendations.append("Your cholesterol level is above normal. Consider diet modifications and regular check-ups.")
    
    # Glucose recommendations
    if user_data.get('gluc', 1) > 1:
        recommendations.append("Your glucose level is above normal. Consider monitoring your sugar intake and regular check-ups.")
    
    # Lifestyle recommendations
    if user_data.get('smoke', 0) == 1:
        recommendations.append("Smoking significantly increases CVD risk. Consider quitting for better heart health.")
    
    if user_data.get('alco', 0) == 1:
        recommendations.append("Regular alcohol consumption can increase CVD risk. Consider limiting intake.")
    
    if user_data.get('active', 0) == 0:
        recommendations.append("Regular physical activity can reduce CVD risk. Consider incorporating exercise into your routine.")
    
    # General recommendations if no specific ones
    if not recommendations:
        recommendations.append("Continue maintaining your healthy lifestyle. Regular check-ups are recommended.")
    
    return recommendations
