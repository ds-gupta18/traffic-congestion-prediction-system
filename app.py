import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)


# PAGE STYLE

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
.stApp {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# TITLE

st.markdown("""
<h1 style='
text-align:center;
color:#00FFAA;
font-size:50px;
margin-top:2px;
margin-bottom:0px;
'>
🚦 Traffic Prediction System
</h1>

<h3 style='
text-align:center;
color:white;
margin-bottom:4px;
'>
Machine Learning Based Traffic Congestion Prediction
</h3>
""", unsafe_allow_html=True)


# LOAD DATA

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# PREPROCESSING
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.dayofweek

df = df.drop(['date_time', 'weather_description'], axis=1)


# TRAFFIC LEVEL FUNCTION
def traffic_level(x):
    if x < 2000:
        return "Low"
    elif x < 4000:
        return "Medium"
    else:
        return "High"


df['traffic_level'] = df['traffic_volume'].apply(
    traffic_level
)

# ENCODING
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

df['weather_main'] = le_weather.fit_transform(
    df['weather_main']
)

df['traffic_level'] = le_traffic.fit_transform(
    df['traffic_level']
)


# MODEL

X = df[
    ['temp', 'rain_1h', 'snow_1h',
     'clouds_all', 'weather_main',
     'hour', 'day']
]

y = df['traffic_level']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)


# SIDEBAR UI

st.sidebar.header("⚙️ Enter Details")

temp_celsius = st.sidebar.number_input(
    "Temperature (°C)",
    value=30.0
)

# Celsius → Kelvin
temp = temp_celsius + 273.15

rain_option = st.sidebar.selectbox(
    "Rain Status",
    ["Not Raining", "Raining"]
)

snow_option = st.sidebar.selectbox(
    "Snow Status",
    ["Not Snowing", "Snowing"]
)

# Convert text to numeric
rain = 1 if rain_option == "Raining" else 0
snow = 1 if snow_option == "Snowing" else 0

clouds = st.sidebar.slider(
    "Cloud %",
    0,
    100
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Clouds", "Clear",
     "Rain", "Mist",
     "Snow"]
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23
)

day = st.sidebar.selectbox(
    "Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day = day_map[day]

predict_btn = st.sidebar.button(
    "🚀 Predict Traffic"
)


# PREDICTION

if predict_btn:

    weather_encoded = le_weather.transform(
        [weather]
    )[0]

    prediction = model.predict([
        [
            temp,
            rain,
            snow,
            clouds,
            weather_encoded,
            hour,
            day
        ]
    ])

    result = le_traffic.inverse_transform(
        prediction
    )

    # CONFIDENCE SCORE
    prediction_proba = model.predict_proba([
        [
            temp,
            rain,
            snow,
            clouds,
            weather_encoded,
            hour,
            day
        ]
    ])

    confidence = max(
        prediction_proba[0]
    ) * 100

    st.markdown("""
    <h2 style='
    text-align:center;
    color:white;
    margin-top:5px;
    '>
    🚦 Traffic Prediction Result
    </h2>
    """, unsafe_allow_html=True)

    # RESULT COLORS

    if result[0] == "Low":
        bg_color = "#28A745"
        emoji = "🟢"

    elif result[0] == "Medium":
        bg_color = "#FFC107"
        emoji = "🟡"

    else:
        bg_color = "#DC3545"
        emoji = "🔴"

    # RESULT BOX

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:8px;
            border-radius:20px;
            text-align:center;
            width:35%;
            margin:auto;
            margin-top:10px;
            box-shadow:0px 0px 20px rgba(255,255,255,0.1);
        ">
            <h1 style="
                color:white;
                font-size:25px;
                font-weight:bold;
            ">
                {emoji} {result[0]} Traffic
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CONFIDENCE SCORE DISPLAY

    st.markdown(
        f"""
        <h3 style='
        text-align:center;
        color:white;
        margin-top:2px;
        margin-bottom:0px;
        font-size:24px;
        '>
        🎯 Confidence:
        {confidence:.2f}%
        </h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # GRAPH 1
    with col1:
        st.subheader("📈 Average Traffic by Hour")

        hourly_traffic = df.groupby(
            'hour'
        )['traffic_volume'].mean()

        st.line_chart(hourly_traffic)


    # GRAPH 2
    with col2:
        st.subheader("📊 Feature Importance")

        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance':
            model.feature_importances_
        })

        importance = importance.sort_values(by='Importance', ascending=False
        )

        st.bar_chart(
            importance.set_index('Feature')
        )


# MODEL ACCURACY

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <h3 style='
        text-align:center;
        color:#00FFAA;
        margin-top:-10px;
        '>
        ✅ Model Accuracy:
        {accuracy * 100:.2f}%
        </h3>
        """,
        unsafe_allow_html=True
    )