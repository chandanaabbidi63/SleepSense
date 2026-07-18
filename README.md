# SleepSense - AI-Powered Sleep Quality Predictor

SleepSense is a machine learning-based web application that predicts sleep quality based on various health and lifestyle factors. It provides personalized suggestions to help users improve their sleep quality.

## Features

- **Sleep Quality Prediction**: Predicts sleep quality on a scale of 1-10
- **Personalized Suggestions**: Provides tailored recommendations based on individual health metrics
- **User-Friendly Interface**: Clean, responsive web interface with real-time predictions
- **Multiple Input Parameters**: Considers age, gender, occupation, sleep duration, physical activity, stress level, BMI, blood pressure, heart rate, and daily steps

## Project Structure

```
SleepSense/
├── Sleep_health_and_lifestyle_dataset.csv  # Training dataset
├── preprocess.py                            # Data preprocessing module
├── train.py                                 # Model training script
├── predict.py                               # Prediction logic with suggestions
├── app.py                                   # Flask web application
├── requirements.txt                         # Python dependencies
├── templates/
│   └── index.html                           # Web interface
├── models/
│   ├── lstm_model.keras                     # (Alternative model)
│   └── notebooks/
│       └── Data_Analysis.ipynb              # Jupyter notebook for analysis
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chandanaabbidi63/decision_tree.git
cd SleepSense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the Random Forest model using the dataset:

```bash
python train.py
```

This will:
- Load and preprocess the sleep health dataset
- Train a Random Forest Regressor model
- Evaluate model performance
- Save the trained model as `sleep_quality_model.pkl`
- Display feature importance

### 2. Run the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### 3. Make Predictions

The app will automatically train the model if it doesn't exist, then start the web server. Users can input their health data through the web form and receive:

- Sleep quality score (1-10)
- Personalized suggestions for improvement

## Input Parameters

The prediction model accepts the following inputs:

| Parameter | Description | Range |
|-----------|-------------|-------|
| Gender | Male/Female | - |
| Age | User's age | 1-120 |
| Occupation | Job type | See dropdown |
| Sleep Duration | Hours of sleep per night | 1-12 |
| Physical Activity Level | Minutes per day | 0-300 |
| Stress Level | Self-reported stress | 1-10 |
| BMI Category | Body mass index category | Normal/Normal Weight/Overweight/Obese |
| Blood Pressure | Systolic/Diastolic (e.g., 120/80) | Format: XXX/XX |
| Heart Rate | Beats per minute | 40-200 |
| Daily Steps | Steps walked per day | 0-50000 |
| Sleep Disorder | None/Sleep Apnea/Insomnia | - |

## Model Details

- **Algorithm**: Random Forest Regressor
- **n_estimators**: 100
- **max_depth**: 20
- **Features**: 11 input parameters
- **Target**: Sleep Quality (1-10 scale)

### Preprocessing Steps

1. Label encoding for categorical variables (Gender, Occupation, BMI Category, Sleep Disorder)
2. Blood pressure splitting into Systolic and Diastolic components
3. Standard scaling of all numerical features
4. Missing value handling
5. Feature selection (excluding Person ID)

## Suggestions Logic

The app provides personalized suggestions based on:

- Sleep quality score category (Excellent/Good/Fair/Poor)
- Individual parameter analysis:
  - Sleep duration optimization
  - Physical activity improvement
  - Stress management
  - BMI awareness
  - Heart rate monitoring
  - Daily activity level
- General sleep hygiene tips

## Example Output

```
Predicted Sleep Quality: 6/10

Good 😊

Personalized Suggestions:
• Consider increasing exercise to 60+ minutes for better sleep
• Monitor stress levels and consider relaxation techniques
• Try to walk at least 7,000-10,000 steps daily

General Sleep Improvement Tips:
• Maintain a consistent sleep schedule
• Avoid caffeine 4-6 hours before bedtime
• Create a dark, quiet sleeping environment
```

## Testing

Test the prediction system with sample data:

```bash
python predict.py
```

This will run a test prediction using sample user data.

## Future Improvements

- [ ] Add LSTM model for time-series sleep data
- [ ] Implement sleep pattern tracking over time
- [ ] Add sleep stage prediction (REM, Light, Deep)
- [ ] Integrate with wearable devices
- [ ] Add sleep diary feature
- [ ] Implement sleep recommendation engine
- [ ] Add progressive web app (PWA) support
- [ ] Multi-language support

## Dataset

The model is trained on the "Sleep Health and Lifestyle Dataset" which contains data about:
- Sleep patterns and quality
- Physical activity levels
- Stress levels
- Health metrics (BMI, blood pressure, heart rate)
- Sleep disorders

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is open source and available under the MIT License.

## Author

**Chandana Abbidi**

## Technologies Used

- Python 3.x
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas / NumPy (Data Processing)
- HTML/CSS/JavaScript (Frontend)

## Contact

For questions or feedback, please open an issue on GitHub.