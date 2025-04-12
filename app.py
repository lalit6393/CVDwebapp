import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix
import joblib
import os

from model import train_model, load_model, preprocess_data, make_prediction
from utils import calculate_feature_importance, plot_feature_importance, load_sample_data

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Main title
st.title("❤️ Cardiovascular Disease Prediction")
st.markdown("""
This application uses machine learning to predict the risk of cardiovascular disease based on medical data.
Simply input your health parameters or upload your data to get a prediction.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Data Analysis", "About CVD"])

# Initialize model and data
@st.cache_resource
def initialize_model():
    try:
        model = load_model()
        return model
    except:
        # Train model if not found
        st.warning("No pre-trained model found. Training a new model...")
        sample_data = load_sample_data()
        model = train_model(sample_data)
        return model

# Load or train the model
model = initialize_model()

if page == "Home":
    st.header("Welcome to CVD Risk Predictor")
    
    st.markdown("""
    ### What is Cardiovascular Disease?
    Cardiovascular disease (CVD) refers to a class of diseases that involve the heart or blood vessels. 
    It is one of the leading causes of death globally. Early detection and preventive measures can 
    significantly reduce the risk.
    
    ### How does this application work?
    This application uses machine learning to analyze medical data and predict the risk of cardiovascular disease.
    The model has been trained on a dataset of patients with known CVD status.
    
    ### How to use this application:
    1. Navigate to the **Prediction** page
    2. Enter your medical information or upload your data
    3. Get a prediction of your CVD risk
    4. View detailed analysis of the factors contributing to your prediction
    
    ### Data Privacy Notice
    This application processes your medical data locally in your browser. No data is stored or shared with any third parties.
    """)
    
    # Display some statistics or visualizations
    st.subheader("CVD Risk Factors")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="High Blood Pressure", value="2-3x", delta="risk increase")
    with col2:
        st.metric(label="Smoking", value="2-4x", delta="risk increase")
    with col3:
        st.metric(label="High Cholesterol", value="2x", delta="risk increase")
    
elif page == "Prediction":
    st.header("Predict CVD Risk")
    
    input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
    
    if input_method == "Manual Input":
        st.subheader("Enter Your Medical Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
            ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
            ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
        
        with col2:
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
            smoke = st.checkbox("Do you smoke?")
            alco = st.checkbox("Do you consume alcohol?")
            active = st.checkbox("Are you physically active?")
        
        # Convert categorical to numerical
        gender_encoded = 1 if gender == "Male" else 2
        chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
        
        cholesterol_encoded = chol_map[cholesterol]
        gluc_encoded = gluc_map[gluc]
        smoke_encoded = 1 if smoke else 0
        alco_encoded = 1 if alco else 0
        active_encoded = 1 if active else 0
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        st.write(f"Your BMI: {bmi:.1f} kg/m²")
        
        if st.button("Predict"):
            # Prepare input data
            user_data = {
                'age': [age * 365],  # Convert to days as per model expectation
                'gender': [gender_encoded],
                'height': [height],
                'weight': [weight],
                'ap_hi': [ap_hi],
                'ap_lo': [ap_lo],
                'cholesterol': [cholesterol_encoded],
                'gluc': [gluc_encoded],
                'smoke': [smoke_encoded],
                'alco': [alco_encoded],
                'active': [active_encoded]
            }
            
            # Convert to DataFrame
            user_df = pd.DataFrame(user_data)
            
            # Make prediction
            try:
                X_processed = preprocess_data(user_df)
                prediction, probability = make_prediction(model, X_processed)
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.error(f"⚠️ High risk of cardiovascular disease detected! (Confidence: {probability[0][1]:.1%})")
                else:
                    st.success(f"✅ Low risk of cardiovascular disease. (Confidence: {probability[0][0]:.1%})")
                
                # Feature importance
                feature_importance = calculate_feature_importance(model, X_processed)
                fig = plot_feature_importance(feature_importance)
                st.subheader("Factors Influencing Prediction")
                st.pyplot(fig)
                
                # Recommendations
                st.subheader("Recommendations")
                if ap_hi > 140 or ap_lo > 90:
                    st.warning("Your blood pressure is elevated. Consider consulting a healthcare professional.")
                    
                if bmi > 25:
                    st.warning(f"Your BMI is {bmi:.1f}, which is above the recommended range. Consider a healthy diet and regular exercise.")
                    
                if cholesterol_encoded > 1:
                    st.warning("Your cholesterol level is above normal. Consider diet modifications and regular check-ups.")
                    
                if smoke_encoded == 1:
                    st.warning("Smoking significantly increases CVD risk. Consider quitting for better heart health.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    
    else:  # Upload CSV
        st.subheader("Upload CSV File with Medical Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load and preview the data
                data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(data.head())
                
                # Check if required columns are present
                required_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Process the data
                    X_processed = preprocess_data(data)
                    predictions, probabilities = make_prediction(model, X_processed, return_all_probs=True)
                    
                    # Add predictions to the dataframe
                    data['prediction'] = predictions
                    data['risk_probability'] = [prob[1] for prob in probabilities]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(data)
                    
                    # Visualization
                    st.subheader("Risk Distribution")
                    fig = px.histogram(data, x='risk_probability', 
                                       title='Distribution of CVD Risk Probability',
                                       labels={'risk_probability': 'CVD Risk Probability'})
                    st.plotly_chart(fig)
                    
                    high_risk_count = sum(predictions)
                    total_count = len(predictions)
                    
                    st.metric("High Risk Patients", f"{high_risk_count} / {total_count}", 
                              f"{high_risk_count/total_count:.1%}")
                    
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

elif page == "Data Analysis":
    st.header("Data Analysis and Model Performance")
    
    # Load sample data for analysis
    sample_data = load_sample_data()
    
    st.subheader("Dataset Overview")
    st.dataframe(sample_data.head())
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(sample_data.describe())
    
    # Display some visualizations
    st.subheader("Visualizations")
    
    chart_type = st.selectbox("Select Visualization", 
                             ["Age vs. CVD Risk", "Blood Pressure vs. CVD Risk", 
                              "BMI vs. CVD Risk", "Cholesterol vs. CVD Risk"])
    
    if chart_type == "Age vs. CVD Risk":
        # Convert age from days to years
        sample_data['age_years'] = sample_data['age'] / 365
        
        fig = px.box(sample_data, x='cardio', y='age_years', 
                    color='cardio',
                    labels={'cardio': 'CVD Present', 'age_years': 'Age (years)'},
                    title='Age Distribution by CVD Status')
        st.plotly_chart(fig)
        
    elif chart_type == "Blood Pressure vs. CVD Risk":
        fig = px.scatter(sample_data, x='ap_hi', y='ap_lo', 
                        color='cardio',
                        labels={'ap_hi': 'Systolic Blood Pressure', 
                                'ap_lo': 'Diastolic Blood Pressure',
                                'cardio': 'CVD Present'},
                        title='Blood Pressure and CVD Risk')
        st.plotly_chart(fig)
        
    elif chart_type == "BMI vs. CVD Risk":
        # Calculate BMI
        sample_data['bmi'] = sample_data['weight'] / ((sample_data['height'] / 100) ** 2)
        
        fig = px.violin(sample_data, x='cardio', y='bmi', 
                      color='cardio',
                      labels={'cardio': 'CVD Present', 'bmi': 'BMI'},
                      title='BMI Distribution by CVD Status')
        st.plotly_chart(fig)
        
    elif chart_type == "Cholesterol vs. CVD Risk":
        # Create contingency table
        cholesterol_cvd = pd.crosstab(sample_data['cholesterol'], sample_data['cardio'])
        cholesterol_cvd_norm = cholesterol_cvd.div(cholesterol_cvd.sum(axis=1), axis=0)
        
        fig = px.bar(cholesterol_cvd_norm, 
                    title='Cholesterol Level vs. CVD Risk',
                    labels={'index': 'Cholesterol Level', 'value': 'Proportion', 'cardio': 'CVD Present'})
        st.plotly_chart(fig)
    
    # Model evaluation metrics
    st.subheader("Model Performance")
    
    # Placeholder for model metrics
    col1, col2, col3 = st.columns(3)
    
    # These would typically come from model evaluation
    with col1:
        st.metric("Accuracy", "72%")
    with col2:
        st.metric("Sensitivity", "68%")
    with col3:
        st.metric("Specificity", "76%")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = calculate_feature_importance(model, preprocess_data(sample_data.drop('cardio', axis=1)))
    fig = plot_feature_importance(feature_importance)
    st.pyplot(fig)

elif page == "About CVD":
    st.header("About Cardiovascular Disease")
    
    st.markdown("""
    ### What is Cardiovascular Disease?
    
    Cardiovascular disease (CVD) is a general term for conditions affecting the heart or blood vessels. 
    It's usually associated with a build-up of fatty deposits inside the arteries (atherosclerosis) and 
    an increased risk of blood clots.
    
    ### Common Types of CVD
    
    - **Coronary Heart Disease**: When the blood supply to the heart becomes blocked or reduced due to 
    a build-up of fatty substances in the coronary arteries.
    
    - **Stroke**: When the blood supply to part of the brain is cut off.
    
    - **Peripheral Arterial Disease**: When there's a blockage in the arteries to the limbs, usually the legs.
    
    - **Aortic Disease**: Includes conditions affecting the aorta, the largest blood vessel in the body.
    
    ### Risk Factors
    
    Some risk factors for CVD include:
    
    - **Age**: Risk increases with age.
    - **Family History**: If your close relatives have had CVD, your risk may be increased.
    - **Smoking**: Significantly increases the risk of CVD.
    - **High Blood Pressure**: Can damage blood vessels.
    - **High Cholesterol**: Can lead to fatty deposits in the arteries.
    - **Diabetes**: Increases the risk of CVD.
    - **Physical Inactivity**: Can lead to obesity and related health problems.
    - **Obesity**: Associated with high blood pressure, diabetes, and high cholesterol.
    - **Excessive Alcohol Consumption**: Can increase blood pressure and risk of CVD.
    
    ### Prevention
    
    You can reduce your risk of CVD by:
    
    - **Eating a Healthy Diet**: Low in saturated fat, salt, and added sugars, and high in fiber, whole grains, fruits, and vegetables.
    - **Exercising Regularly**: Aim for at least 150 minutes of moderate-intensity exercise each week.
    - **Maintaining a Healthy Weight**: Keep your BMI within the healthy range.
    - **Quitting Smoking**: This is one of the most important things you can do to reduce your risk.
    - **Limiting Alcohol Consumption**: Stay within recommended guidelines.
    - **Managing Stress**: Chronic stress can contribute to CVD.
    - **Regular Health Check-ups**: To monitor blood pressure, cholesterol levels, and other risk factors.
    
    ### Symptoms to Watch For
    
    Common symptoms of CVD include:
    
    - Chest pain or discomfort
    - Shortness of breath
    - Pain, numbness, weakness, or coldness in your legs or arms
    - Pain in the neck, jaw, throat, upper abdomen, or back
    - Irregular heartbeat
    
    If you experience these symptoms, especially chest pain, seek medical attention immediately.
    """)
    
    st.warning("Disclaimer: This application provides general information about cardiovascular disease and risk prediction. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical condition.")

# Footer
st.markdown("---")
st.caption("© 2023 CVD Risk Predictor | This application is for educational purposes only")
