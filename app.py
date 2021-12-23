import predict_vaccination
from flask import Flask
import pandas as pd
import json

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return {"Welcome": "access to ML prediction to route /api/predict_json"}


@app.route('/api/predict_json', methods=['GET'])
def predict_json():
    df_forecast, df_original = predict_vaccination.predict()

    df_forecast.loc[:, ('date')] = df_forecast['date'].dt.strftime('%Y-%m-%d')
    df_original.loc[:, ('date')] = df_original['date'].dt.strftime('%Y-%m-%d')

    prediction_json = df_forecast.to_json(orient='records').strip("\'")
    original_json = df_original.to_json(orient='records').strip("\'")

    return {"prediction": json.loads(prediction_json),
            "original": json.loads(original_json),
            "total": json.loads(original_json) + json.loads(prediction_json)}


if __name__ == "__main__":
    app.run(debug=True)
