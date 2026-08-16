from flask import Flask, request, jsonify
import joblib
from pathlib import Path

from app.utils import prepare_customer_data


app = Flask(__name__)


# Find the directory containing this file
APP_DIR = Path(__file__).resolve().parent


# Load the saved transformer and model
transformer = joblib.load(APP_DIR / "transformer.pkl")
model = joblib.load(APP_DIR / "model.pkl")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "customer" not in data:
            return jsonify({
                "error": "Request must contain a 'customer' object."
            }), 400

        customer = data["customer"]

        customer_df = prepare_customer_data(customer)

        transformed_customer = transformer.transform(customer_df)

        churn_probability = model.predict_proba(
            transformed_customer
        )[0][1]

        churn_prediction = (
            "Yes"
            if churn_probability >= 0.5
            else "No"
        )

        return jsonify({
            "churn_probability": float(churn_probability),
            "churn_prediction": churn_prediction
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000
    )