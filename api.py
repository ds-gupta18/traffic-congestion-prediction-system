from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Preprocessing
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.dayofweek
df = df.drop(['date_time', 'weather_description'], axis=1)

def traffic_level(x):
    if x < 2000:
        return 0
    elif x < 4000:
        return 1
    else:
        return 2

df['traffic_level'] = df['traffic_volume'].apply(traffic_level)

le_weather = LabelEncoder()
df['weather_main'] = le_weather.fit_transform(df['weather_main'])

X = df[['temp','rain_1h','snow_1h','clouds_all','weather_main','hour','day']]
y = df['traffic_level']

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return "API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    weather = le_weather.transform([data['weather']])[0]

    input_data = [[
        data['temp'], data['rain'], data['snow'],
        data['clouds'], weather, data['hour'], data['day']
    ]]

    prediction = model.predict(input_data)[0]

    result = ["Low", "Medium", "High"][prediction]

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)