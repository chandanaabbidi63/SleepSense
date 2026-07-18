import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def load_and_preprocess_data(csv_path):
    """Load and preprocess the sleep dataset for GRU"""
    df = pd.read_csv(csv_path)
    
    # Drop Person ID as it's not needed
    df = df.drop('Person ID', axis=1)
    
    # Handle missing values if any
    df = df.dropna()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi = LabelEncoder()
    le_sleep_disorder = LabelEncoder()
    
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
    df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = le_sleep_disorder.fit_transform(df['Sleep Disorder'])
    
    # Split Blood Pressure into Systolic and Diastolic
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop('Blood Pressure', axis=1)
    
    # Separate features and target
    X = df.drop('Quality of Sleep', axis=1)
    y = df['Quality of Sleep']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for GRU input [samples, timesteps, features]
    X_gru = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Save encoders and scaler for later use
    with open('encoders.pkl', 'wb') as f:
        pickle.dump({
            'gender': le_gender,
            'occupation': le_occupation,
            'bmi': le_bmi,
            'sleep_disorder': le_sleep_disorder,
            'scaler': scaler
        }, f)
    
    return X_gru, y, X.columns.tolist()

def preprocess_user_input(user_data, encoders_path='encoders.pkl'):
    """Preprocess user input for GRU prediction"""
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Create DataFrame from user input
    df = pd.DataFrame([user_data])
    
    # Handle Blood Pressure
    blood_pressure = df['Blood Pressure'].iloc[0]
    print(f"Blood Pressure value: {blood_pressure}")
    if blood_pressure and blood_pressure != 'None':
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    else:
        df['Systolic_BP'] = 120.0
        df['Diastolic_BP'] = 80.0
    df = df.drop('Blood Pressure', axis=1)
    
    # Encode categorical variables with handling for unseen labels
    df['Gender'] = encoders['gender'].transform(df['Gender'])
    
    # Handle Occupation - map unseen values to most similar known category
    occupation = df['Occupation'].iloc[0]
    if occupation in encoders['occupation'].classes_:
        df['Occupation'] = encoders['occupation'].transform([occupation])[0]
    else:
        # Map unseen occupation to closest match or default
        occupation_mapping = {
            'Student': 'Engineer',  # Map Student to Engineer as default
            'Teacher': 'Teacher',
            'Doctor': 'Doctor',
            'Nurse': 'Nurse',
            'Engineer': 'Engineer',
            'Software Engineer': 'Software Engineer',
            'Accountant': 'Accountant',
            'Lawyer': 'Lawyer',
            'Manager': 'Manager',
            'Sales Representative': 'Sales Representative'
        }
        mapped_occ = occupation_mapping.get(occupation, 'Engineer')
        df['Occupation'] = encoders['occupation'].transform([mapped_occ])[0]
    
    df['BMI Category'] = encoders['bmi'].transform(df['BMI Category'])
    
    # Handle Sleep Disorder - map unseen values
    sleep_disorder = df['Sleep Disorder'].iloc[0]
    if sleep_disorder in encoders['sleep_disorder'].classes_:
        df['Sleep Disorder'] = encoders['sleep_disorder'].transform([sleep_disorder])[0]
    else:
        # Map to 'None' if unseen
        df['Sleep Disorder'] = encoders['sleep_disorder'].transform(['None'])[0]
    
    # Scale features
    X = encoders['scaler'].transform(df)
    
    # Reshape for GRU input [samples, timesteps, features]
    X_gru = X.reshape(X.shape[0], 1, X.shape[1])
    
    return X_gru

if __name__ == "__main__":
    X, y, features = load_and_preprocess_data('Sleep_health_and_lifestyle_dataset.csv')
    print(f"Features: {features}")
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")