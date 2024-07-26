from flask import Flask, render_template
from datetime import datetime, timedelta, date

app = Flask(__name__)

def get_rain_predictions():
    # Dummy rain prediction data
    today = date.today()
    return [
        {"date": today, "prediction": "Rainy"},
        {"date": today + timedelta(days=1), "prediction": "Cloudy"},
        {"date": today + timedelta(days=2), "prediction": "Sunny"},
    ]

@app.route('/')
def index():
    predictions = get_rain_predictions()
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)