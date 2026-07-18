import sys
sys.path.insert(0, '.')

from predict import predict_sleep_quality, get_suggestions

# Test with exact user input that was failing
user_data = {
    'Gender': 'Female',
    'Age': 21,
    'Occupation': 'Student',  # This was failing before
    'Sleep Duration': 7.0,
    'Physical Activity Level': 0,
    'Stress Level': 7,
    'BMI Category': 'Normal',
    'Blood Pressure': '110/60',
    'Heart Rate': 60,
    'Daily Steps': 2000,
    'Sleep Disorder': 'None'  # This was failing before
}

try:
    print("Testing with user's exact input (Student + None)...")
    quality = predict_sleep_quality(user_data)
    suggestions = get_suggestions(quality, user_data)
    
    print(f"✓ SUCCESS!")
    print(f"Sleep Quality Score: {quality}/10\n")
    print("Personalized Suggestions:")
    print(suggestions)
    
except Exception as e:
    print(f"✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()