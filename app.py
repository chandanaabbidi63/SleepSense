import streamlit as st
from predict import predict_sleep_quality, get_suggestions
import os

st.set_page_config(
    page_title="SleepSense AI",
    page_icon="😴",
    layout="wide"
)

st.title("😴 SleepSense AI")
st.markdown("### Predict Your Sleep Quality Using GRU Deep Learning Model")

# Check required files
required_files = [
    "gru_sleep_model.h5",
    "encoders.pkl"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"Missing files: {', '.join(missing)}")
    st.stop()

# -------------------------
# INPUT FORM
# -------------------------

with st.form("sleep_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Student",
                "Engineer",
                "Doctor",
                "Nurse",
                "Teacher",
                "Lawyer",
                "Accountant",
                "Manager",
                "Sales Representative",
                "Software Engineer"
            ]
        )

        sleep_duration = st.slider(
            "Sleep Duration (hours)",
            3.0,
            12.0,
            7.0
        )

        physical_activity = st.slider(
            "Physical Activity (minutes/day)",
            0,
            180,
            45
        )

        stress_level = st.slider(
            "Stress Level",
            1,
            10,
            5
        )

    with col2:

        bmi_category = st.selectbox(
            "BMI Category",
            [
                "Normal",
                "Overweight",
                "Obese"
            ]
        )

        systolic = st.number_input(
            "Systolic BP",
            80,
            220,
            120
        )

        diastolic = st.number_input(
            "Diastolic BP",
            40,
            140,
            80
        )

        heart_rate = st.number_input(
            "Heart Rate",
            40,
            180,
            72
        )

        daily_steps = st.number_input(
            "Daily Steps",
            0,
            30000,
            6000
        )

        sleep_disorder = st.selectbox(
            "Sleep Disorder",
            [
                "None",
                "Sleep Apnea",
                "Insomnia"
            ]
        )

    submitted = st.form_submit_button("🔮 Predict Sleep Quality")

# -------------------------
# PREDICTION
# -------------------------

if submitted:

    user_data = {

        "Gender": gender,

        "Age": age,

        "Occupation": occupation,

        "Sleep Duration": sleep_duration,

        "Physical Activity Level": physical_activity,

        "Stress Level": stress_level,

        "BMI Category": bmi_category,

        "Blood Pressure": f"{systolic}/{diastolic}",

        "Heart Rate": heart_rate,

        "Daily Steps": daily_steps,

        "Sleep Disorder": sleep_disorder
    }

    try:

        with st.spinner("Predicting..."):

            quality = predict_sleep_quality(user_data)

            suggestions = get_suggestions(
                quality,
                user_data
            )

        st.success("Prediction Completed!")

        st.metric(
            "Predicted Sleep Quality",
            f"{quality}/10"
        )

        if quality >= 8:

            st.success("🌟 Excellent Sleep!")

        elif quality >= 6:

            st.info("🙂 Good Sleep")

        elif quality >= 4:

            st.warning("😐 Fair Sleep")

        else:

            st.error("😴 Poor Sleep")

        st.subheader("📋 Personalized Suggestions")

        st.text(suggestions)

    except Exception as e:

        st.error(f"Prediction Failed\n\n{e}")