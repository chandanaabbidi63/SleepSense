import sys
sys.path.insert(0, '.')

from predict import predict_sleep_quality, get_suggestions
from preprocess import preprocess_user_input

# Test with actual user-provided data
user_data = {
    'Gender': 'Male',
    'Age': 30,
    'Occupation': 'Software Engineer',
    'Sleep Duration': 7.0,
    'Physical Activity Level': 0,
    'Stress Level': 6,
    'BMI Category': 'Normal',
    'Blood Pressure': '110/70',
    'Heart Rate': 70,
    'Daily Steps': 2000,
    'Sleep Disorder': 'None'
}

try:
    print("Testing GRU preprocessing...")
    X = preprocess_user_input(user_data)
    print(f"✓ Preprocessing successful! Shape: {X.shape}")
    
    print("\nTesting GRU prediction...")
    quality = predict_sleep_quality(user_data)
    print(f"✓ Prediction successful! Score: {quality}/10")
    
    print("\nGetting suggestions...")
    suggestions = get_suggestions(quality, user_data)
    
    print("\n" + "="*60)
    print(f"FINAL RESULT:")
    print(f"Sleep Quality Score: {quality}/10\n")
    print("Personalized Suggestions:")
    print(suggestions)
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()