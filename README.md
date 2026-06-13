# 🚦Traffic Congestion Prediction System
A Machine Learning-powered Traffic Congestion Prediction System built using Python, Streamlit, and Scikit-Learn. The application predicts traffic congestion levels based on weather conditions, cloud coverage, time, and day information.
The project provides an interactive dashboard featuring real-time predictions, confidence scores, traffic trend visualization, and feature importance analysis.

## Project Overview
Traffic congestion is one of the major challenges in urban areas, leading to delays, increased fuel consumption, and environmental concerns.
This project leverages Machine Learning techniques to analyze historical traffic and weather data and predict traffic congestion levels. Users can input weather and time-related parameters through an interactive dashboard and receive instant traffic predictions.

The system classifies traffic into three categories:
- 🟢 Low Traffic
- 🟡 Medium Traffic
- 🔴 High Traffic

using a Random Forest Classifier.

## Features
- 🚦 Real-Time Traffic Prediction
- 🎯 Prediction Confidence Score
- 📈 Average Traffic by Hour Visualization
- 📊 Feature Importance Analysis
- 🌙 Modern Dark-Themed User Interface
- ⚡ Interactive Streamlit Dashboard
- 🤖 Machine Learning-Based Traffic Classification

## Tech Stack

# Programming Language
- Python

# Machine Learning
- Scikit-Learn
- Random Forest Classifier

# Data Processing
- Pandas
- NumPy

# Web Framework
- Streamlit

## Dataset
- Metro Interstate Traffic Volume Dataset

## Project Structure
traffic-congestion-prediction-system/
│
├── app.py
├── requirements.txt
├── Metro_Interstate_Traffic_Volume.csv
├── Traffic congestion documentation.pdf
├── .gitignore
└── README.md

##  Machine Learning Workflow

# 1. Data Collection
The project uses the Metro Interstate Traffic Volume Dataset containing traffic and weather-related information.

# 2. Data Preprocessing
The following preprocessing steps are performed:
- Date-time conversion
- Hour extraction
- Day extraction
- Label Encoding
- Feature selection

# 3. Traffic Categorization
Traffic volume is converted into categories:

| Traffic Volume | Category |
|---------------|-----------|
| Less than 2000 | Low |
| 2000 to 4000 | Medium |
| Greater than 4000 | High |

# 4. Model Training
The Random Forest Classifier is trained using:

python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# 5. Prediction
The model predicts traffic based on:
- Temperature
- Rain Status
- Snow Status
- Cloud Percentage
- Weather Condition
- Hour
- Day

## Input Features

| Feature | Description |
|----------|------------|
| Temperature | Current temperature |
| Rain Status | Whether it is raining |
| Snow Status | Whether it is snowing |
| Cloud % | Cloud coverage percentage |
| Weather | Weather condition |
| Hour | Current hour |
| Day | Day of the week |

## Dashboard Visualizations

# Average Traffic by Hour
Displays average traffic volume across different hours of the day.

# Feature Importance
Shows the contribution of each feature to the prediction model.

# Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 92.24% |

The model achieves approximately 92% accuracy on the testing dataset.

##  Installation & Setup

# Clone Repository
```bash
git clone https://github.com/ds-gupta18/traffic-congestion-prediction-system.git
```

# Navigate to Project Folder
```bash
cd traffic-congestion-prediction-system
```

# Install Dependencies
```bash
pip install -r requirements.txt
```

# Run the Application
```bash
streamlit run app.py
```

##  Application Preview

# Dashboard Features
- Real-Time Traffic Prediction
- Confidence Score Display
- Average Traffic Trend Graph
- Feature Importance Graph
- Dark-Themed Interactive UI

# Future Enhancements
- Live Traffic API Integration
- Route Optimization
- Traffic Forecasting
- Interactive Maps
- Deep Learning-Based Prediction Models
- AWS Cloud Deployment


---

## 📜 License

This project is developed for educational and learning purposes.
