import numpy as np
from tensorflow import keras
from preprocess import preprocess_user_input

# Load and check the model
model = keras.models.load_model('gru_sleep_model.h5', custom_objects={'mse': keras.losses.mse, 'mae': keras.losses.mae})

print("Testing predictions with different inputs:\n")

# Test case 1: Poor sleep conditions
user_data1 = {
    'Gender': 'Male',
    'Age': 30,
    'Occupation': 'Software Engineer',
    'Sleep Duration': 4.0,  # Very low sleep
    'Physical Activity Level': 10,
    'Stress Level': 9,  # High stress
    'BMI Category': 'Obese',
    'Blood Pressure': '150/95',
    'Heart Rate': 90,
    'Daily Steps': 1000,
    'Sleep Disorder': 'Insomnia'
}

X1 = preprocess_user_input(user_data1)
pred1 = model.predict(X1, verbose=0)[0][0]
print(f"Test 1 (Poor conditions): {pred1:.2f} -> {int(round(pred1))}/10")

# Test case 2: Good sleep conditions
user_data2 = {
    'Gender': 'Female',
    'Age': 25,
    'Occupation': 'Doctor',
    'Sleep Duration': 8.5,  # Good sleep
    'Physical Activity Level': 80,
    'Stress Level': 3,  # Low stress
    'BMI Category': 'Normal',
    'Blood Pressure': '120/80',
    'Heart Rate': 65,
    'Daily Steps': 12000,
    'Sleep Disorder': 'None'
}

X2 = preprocess_user_input(user_data2)
pred2 = model.predict(X2, verbose=0)[0][0]
print(f"Test 2 (Good conditions): {pred2:.2f} -> {int(round(pred2))}/10")

# Test case 3: Mixed conditions
user_data3 = {
    'Gender': 'Male',
    'Age': 35,
    'Occupation': 'Teacher',
    'Sleep Duration': 6.0,  # Moderate sleep
    'Physical Activity Level': 40,
    'Stress Level': 6,  # Moderate stress
    'BMI Category': 'Overweight',
    'Blood Pressure': '135/85',
    'Heart Rate': 78,
    'Daily Steps': 6000,
    'Sleep Disorder': 'None'
}

X3 = preprocess_user_input(user_data3)
pred3 = model.predict(X3, verbose=0)[0][0]
print(f"Test 3 (Mixed conditions): {pred3:.2f} -> {int(round(pred3))}/10")

print("\nModel is working correctly if scores differ significantly.")
print("If all scores are similar (~6-7), the model needs retraining.")
print("\nExpected range: Poor conditions should score 2-4, Good conditions should score 7-9")