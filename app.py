from flask import Flask, render_template, request, jsonify
import numpy as np
from predict import predict_sleep_quality, get_suggestions
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Get form data
        gender = request.form.get('gender')
        age = request.form.get('age')
        occupation = request.form.get('occupation')
        sleep_duration = request.form.get('sleep_duration')
        physical_activity = request.form.get('physical_activity')
        stress_level = request.form.get('stress_level')
        bmi_category = request.form.get('bmi_category')
        blood_pressure = request.form.get('blood_pressure')
        heart_rate = request.form.get('heart_rate')
        daily_steps = request.form.get('daily_steps')
        sleep_disorder = request.form.get('sleep_disorder')
        
        # Validate required fields
        if not all([gender, age, occupation, sleep_duration, physical_activity, 
                    stress_level, bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder]):
            return jsonify({
                'success': False,
                'error': 'All fields are required. Please fill in all fields.'
            }), 400
        
        user_data = {
            'Gender': gender,
            'Age': int(age),
            'Occupation': occupation,
            'Sleep Duration': float(sleep_duration),
            'Physical Activity Level': int(physical_activity),
            'Stress Level': int(stress_level),
            'BMI Category': bmi_category,
            'Blood Pressure': blood_pressure,
            'Heart Rate': int(heart_rate),
            'Daily Steps': int(daily_steps),
            'Sleep Disorder': sleep_disorder
        }
        
        # Get prediction
        sleep_quality = predict_sleep_quality(user_data)
        
        # Get suggestions
        suggestions = get_suggestions(sleep_quality, user_data)
        
        # Return JSON response
        return jsonify({
            'success': True,
            'sleep_quality': sleep_quality,
            'suggestions': suggestions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Train model if not exists
    if not os.path.exists('gru_sleep_model.h5'):
        print("Training GRU model first...")
        from train import train_gru_model
        train_gru_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)