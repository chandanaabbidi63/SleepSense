import streamlit as st
import numpy as np
from tensorflow import keras
from preprocess import preprocess_user_input
from predict import predict_sleep_quality, get_suggestions
import os

# Page configuration
st.set_page_config(
    page_title="SleepSense - AI Sleep Quality Predictor",
    page_icon="😴",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1.2em;
        font-weight: 700;
        margin: 20px 0;
        cursor: pointer;
        border-radius: 15px;
        width: 100%;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    .score-display {
        font-size: 4em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header with emoji
    st.markdown('<p class="main-header">😴 SleepSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ AI-Powered Sleep Quality Predictor ✨</p>', unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('gru_sleep_model.h5'):
        st.error("⚠️ Model not found. Please run train.py first to train the GRU model.")
        st.stop()
    
    # Create form
    with st.form("sleep_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            age = st.number_input("Age", min_value=1, max_value=120, value=21)
        
        occupation = st.selectbox(
            "Occupation",
            ["Student", "Doctor", "Teacher", "Nurse", "Engineer", 
             "Software Engineer", "Accountant", "Lawyer", "Manager", "Sales Representative"]
        )
        
        st.markdown("### ⏰ Sleep & Lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=1.0, max_value=12.0, value=7.0, step=0.1)
        with col2:
            physical_activity = st.number_input("Physical Activity (min/day)", min_value=0, max_value=300, value=45)
        
        col1, col2 = st.columns(2)
        with col1:
            stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        with col2:
            bmi_category = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"])
        
        st.markdown("### 💓 Health Metrics")
        col1, col2 = st.columns(2)
        with col1:
            blood_pressure = st.text_input("Blood Pressure (systolic/diastolic)", value="120/80", placeholder="120/80")
        with col2:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
        
        col1, col2 = st.columns(2)
        with col1:
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=8000)
        with col2:
            sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Sleep Apnea", "Insomnia"])
        
        # Submit button
        submitted = st.form_submit_button("🌟 Analyze My Sleep Quality")
    
    if submitted:
        try:
            # Prepare user data
            user_data = {
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'Sleep Duration': sleep_duration,
                'Physical Activity Level': physical_activity,
                'Stress Level': stress_level,
                'BMI Category': bmi_category,
                'Blood Pressure': blood_pressure,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps,
                'Sleep Disorder': sleep_disorder
            }
            
            with st.spinner('🔮 Analyzing your sleep quality...'):
                # Get prediction
                sleep_quality = predict_sleep_quality(user_data)
                
                # Get suggestions
                suggestions = get_suggestions(sleep_quality, user_data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Your Results")
            
            # Create metric container
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="score-display">{sleep_quality}/10</div>', unsafe_allow_html=True)
            
            # Quality indicator
            if sleep_quality >= 8:
                st.success(f"🌟 Excellent Sleep Quality! Keep maintaining your healthy habits.")
            elif sleep_quality >= 6:
                st.info(f"😊 Good sleep quality ({sleep_quality}/10)")
            elif sleep_quality >= 4:
                st.warning(f"😐 Fair sleep quality ({sleep_quality}/10)")
            else:
                st.error(f"😞 Poor sleep quality ({sleep_quality}/10)")
            
            # Display suggestions
            st.markdown("### 💡 Personalized Suggestions:")
            st.info(suggestions)
            
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            with st.expander("Show error details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666; padding: 20px;'>Made with ❤️ by SleepSense AI</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()