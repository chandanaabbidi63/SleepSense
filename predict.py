import numpy as np
from tensorflow import keras
from preprocess import preprocess_user_input

def predict_sleep_quality(user_data, model_path='gru_sleep_model.h5'):
    """Predict sleep quality using GRU model"""
    try:
        # Load model with custom objects to handle metric deserialization
        model = keras.models.load_model(model_path, custom_objects={'mse': keras.losses.mse, 'mae': keras.losses.mae})
        
        # Preprocess user input
        X = preprocess_user_input(user_data)
        
        # Make prediction
        prediction = model.predict(X, verbose=0)[0][0]
        
        # Scale prediction more dynamically based on input factors
        sleep_duration = user_data.get('Sleep Duration', 7)
        stress_level = user_data.get('Stress Level', 5)
        physical_activity = user_data.get('Physical Activity Level', 50)
        
        # Adjust prediction based on key factors
        adjustment = 0
        if sleep_duration < 6:
            adjustment -= 1.5
        elif sleep_duration < 7:
            adjustment -= 0.5
        elif sleep_duration >= 8:
            adjustment += 0.5
        
        if stress_level >= 8:
            adjustment -= 1.5
        elif stress_level >= 6:
            adjustment -= 0.5
        elif stress_level <= 3:
            adjustment += 0.5
        
        if physical_activity < 20:
            adjustment -= 0.5
        elif physical_activity >= 60:
            adjustment += 0.5
        
        # Apply adjustment
        adjusted_pred = prediction + adjustment
        
        # Round to nearest integer (sleep quality typically 1-10)
        sleep_quality = int(round(adjusted_pred))
        sleep_quality = max(1, min(10, sleep_quality))  # Clamp between 1 and 10
        
        return sleep_quality
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def get_suggestions(sleep_quality, user_data):
    """Generate personalized suggestions based on sleep quality and user data"""
    suggestions = []
    
    if sleep_quality >= 8:
        suggestions.append("Excellent sleep quality! Keep maintaining your healthy habits.")
    elif sleep_quality >= 6:
        suggestions.append("Good sleep quality. Here are some tips to improve further:")
    elif sleep_quality >= 4:
        suggestions.append("Fair sleep quality. Consider making these changes:")
    else:
        suggestions.append("Poor sleep quality. We recommend taking immediate action:")
    
    # Analyze individual factors
    sleep_duration = user_data.get('Sleep Duration', 7)
    physical_activity = user_data.get('Physical Activity Level', 50)
    stress_level = user_data.get('Stress Level', 5)
    bmi_category = user_data.get('BMI Category', 'Normal')
    heart_rate = user_data.get('Heart Rate', 70)
    daily_steps = user_data.get('Daily Steps', 5000)
    
    # Sleep duration suggestions
    if sleep_duration < 6:
        suggestions.append(f"• Your sleep duration ({sleep_duration}h) is insufficient. Increase to 7-9 hours per night")
    elif sleep_duration > 9:
        suggestions.append(f"• Your sleep duration ({sleep_duration}h) is too long. Aim for 7-9 hours")
    else:
        suggestions.append(f"✓ Your sleep duration ({sleep_duration}h) is optimal")
    
    # Physical activity suggestions
    if physical_activity < 30:
        suggestions.append(f"• Your physical activity ({physical_activity} min) is low. Increase to at least 30 minutes daily")
    elif physical_activity < 60:
        suggestions.append(f"• Your activity ({physical_activity} min) is moderate. Consider 60+ minutes for better sleep")
    else:
        suggestions.append(f"✓ Good physical activity level ({physical_activity} min)")
    
    # Stress management
    if stress_level >= 8:
        suggestions.append(f"• Your stress level ({stress_level}/10) is very high. Practice meditation, yoga, or deep breathing exercises")
        suggestions.append("• Consider reducing work load or seeking professional support")
    elif stress_level >= 6:
        suggestions.append(f"• Your stress level ({stress_level}/10) is elevated. Try relaxation techniques before bed")
    elif stress_level <= 3:
        suggestions.append(f"✓ Excellent stress management ({stress_level}/10)")
    else:
        suggestions.append(f"✓ Stress levels are well-managed ({stress_level}/10)")
    
    # BMI suggestions
    if bmi_category == 'Obese':
        suggestions.append("• Focus on maintaining a healthy weight through proper diet and regular exercise")
        suggestions.append("• Consider consulting a nutritionist for personalized advice")
    elif bmi_category == 'Overweight':
        suggestions.append("• Consider adopting a balanced diet and increasing physical activity")
    else:
        suggestions.append(f"✓ Healthy BMI category ({bmi_category})")
    
    # Heart rate suggestions
    if heart_rate > 85:
        suggestions.append(f"• Your heart rate ({heart_rate} bpm) is elevated. Consider cardiovascular exercise")
    elif heart_rate < 60 and heart_rate >= 40:
        suggestions.append(f"• Your heart rate ({heart_rate} bpm) is on the lower side. Monitor during exercise")
    else:
        suggestions.append(f"✓ Heart rate is within normal range ({heart_rate} bpm)")
    
    # Daily steps suggestion
    if daily_steps < 3000:
        suggestions.append(f"• Your daily activity ({daily_steps} steps) is low. Aim for 7,000-10,000 steps daily")
    elif daily_steps < 5000:
        suggestions.append(f"• Increase daily steps from {daily_steps} to at least 7,000 for better health")
    else:
        suggestions.append(f"✓ Good daily activity level ({daily_steps} steps)")
    
    # General tips
    suggestions.append("\nGeneral Sleep Improvement Tips:")
    suggestions.append("• Maintain a consistent sleep schedule (sleep and wake at same time daily)")
    suggestions.append("• Avoid caffeine and heavy meals 4-6 hours before bedtime")
    suggestions.append("• Create a dark, quiet, and cool sleeping environment (18-22°C)")
    suggestions.append("• Limit screen time 1 hour before bed")
    suggestions.append("• Avoid napping during the day if you have trouble sleeping at night")
    suggestions.append("• Consider relaxation techniques like reading or meditation before bed")
    
    return "\n".join(suggestions)

if __name__ == "__main__":
    # Test with user's data
    user_data = {
        'Gender': 'Female',
        'Age': 21,
        'Occupation': 'Student',
        'Sleep Duration': 7.0,
        'Physical Activity Level': 0,
        'Stress Level': 7,
        'BMI Category': 'Normal',
        'Blood Pressure': '110/60',
        'Heart Rate': 60,
        'Daily Steps': 2000,
        'Sleep Disorder': 'None'
    }
    
    quality = predict_sleep_quality(user_data)
    suggestions = get_suggestions(quality, user_data)
    
    print(f"Predicted Sleep Quality: {quality}/10")
    print("\nSuggestions:")
    print(suggestions)