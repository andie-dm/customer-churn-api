import argparse
import logging
from pathlib import Path

import pandas as pd
import requests


# -----------------------------
# 1. SET UP LOGGING
# -----------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "batch_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------
# 2. API ENDPOINT
# -----------------------------

API_URL = "http://localhost:8000/predict"


# -----------------------------
# 3. BATCH SCORING FUNCTION
# -----------------------------

def score_customers(input_file):

    df = pd.read_csv(input_file)

    results = []
    failures = 0
    probabilities = []

    logging.info("Batch scoring started.")
    logging.info(f"Total customers to score: {len(df)}")

    for _, row in df.iterrows():

        customer = row.to_dict()

        payload = {
            "customer": customer
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:

                prediction = response.json()

                churn_probability = prediction["churn_probability"]
                churn_prediction = prediction["churn_prediction"]

                probabilities.append(churn_probability)

                scored_row = customer.copy()
                scored_row["churn_probability"] = churn_probability
                scored_row["churn_prediction"] = churn_prediction

                results.append(scored_row)

            else:

                failures += 1

                logging.error(
                    f"Prediction failed with status "
                    f"{response.status_code}: {response.text}"
                )

        except Exception as e:

            failures += 1
            logging.exception(
                f"Prediction request failed: {e}"
            )


    # -----------------------------
    # 4. SAVE SCORED CUSTOMERS
    # -----------------------------

    scored_df = pd.DataFrame(results)

    scored_df.to_csv(
        "scored_customers.csv",
        index=False
    )


    # -----------------------------
    # 5. MONITORING METRICS
    # -----------------------------

    total_requests = len(df)

    if probabilities:
        average_probability = sum(probabilities) / len(probabilities)
    else:
        average_probability = 0


    logging.info(f"Total requests: {total_requests}")
    logging.info(f"Failed predictions: {failures}")
    logging.info(
        f"Average churn probability: "
        f"{average_probability:.4f}"
    )

    logging.info("Batch scoring completed.")


    print("Batch scoring complete.")
    print(f"Total requests: {total_requests}")
    print(f"Failures: {failures}")
    print(
        f"Average churn probability: "
        f"{average_probability:.4f}"
    )
    print("Saved scored_customers.csv")
    print("Saved logs/batch_log.txt")


# -----------------------------
# 6. COMMAND LINE INPUT
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to customer CSV file"
    )

    args = parser.parse_args()

    score_customers(args.input)