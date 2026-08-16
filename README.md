# customer-churn-api
Deploying my Machine Learning Model

# Customer Churn Prediction API

## Project Overview

This project deploys a machine learning model that predicts whether a telecom customer is likely to churn.

The project takes a trained Logistic Regression model and turns it into a real-time Flask API. It also includes a batch scoring pipeline that can score multiple customers by sending each customer to the same API endpoint.

The complete workflow includes:

- Customer churn data preprocessing
- Logistic Regression model training
- Saving the trained model and preprocessing transformer
- Real-time prediction using a Flask API
- Batch customer scoring
- Basic monitoring and logging
- A maintenance plan for future model updates

---

## Machine Learning Workflow

The training workflow begins with the `gold_churn_data.csv` dataset.

The target variable is `Churn`, while the remaining customer attributes are used as model features.

The preprocessing step separates the features into numerical and categorical columns.

Numerical features use mean imputation to handle missing values.

Categorical features are converted into numerical features using One-Hot Encoding.

The preprocessing transformer is fitted to the training data and saved as:

`app/transformer.pkl`

A Logistic Regression classifier is then trained using the transformed customer data and saved as:

`app/model.pkl`

Saving both objects allows the API to make predictions without retraining the model every time the application starts.

---

## Real-Time Prediction API

The Flask application is located in:

`app/main.py`

The API exposes the following endpoint:

`POST /predict`

The endpoint receives customer information as JSON.

The prediction workflow is:

Customer JSON → Pandas DataFrame → Saved Transformer → Trained Model → Churn Prediction

The transformer prepares new customer data using the same preprocessing rules learned during training.

The trained Logistic Regression model then calculates the customer's churn probability.

A probability of 0.50 or greater is classified as:

`Yes`

A probability below 0.50 is classified as:

`No`

The API returns both the churn probability and the final churn prediction.

Example response:

```json
{
    "churn_probability": 0.6410,
    "churn_prediction": "Yes"
}

---

## Model Maintenance Plan

To keep the customer churn model reliable over time, I would regularly monitor both the model predictions and the performance of the deployment pipeline.

### Retraining Strategy

I would review the model periodically using newly collected customer data and actual churn outcomes. I would consider retraining the model every 3–6 months, or sooner if there is evidence that its prediction performance has declined. Retraining may also be necessary when the company introduces major changes such as new services, pricing plans, contracts, or customer policies because these changes could affect customer behavior. Before replacing the existing model, I would evaluate the new version and confirm that it performs appropriately on recent data.

### Drift and Failure Monitoring

I would monitor the batch scoring logs for changes in prediction patterns and technical failures. Important metrics include the number of prediction requests, failed requests, and average churn probability. A significant change in the average churn probability over time could indicate data drift, meaning that current customer behavior is becoming different from the data used to train the model. I would also investigate increases in API errors, missing values, unexpected categories, or failed predictions because these could indicate problems with incoming data or the deployment pipeline.

### Versioning and Documentation

Future versions of the model, preprocessing transformer, and application code should be versioned and documented in GitHub. Each update should use a clear commit message explaining what changed, such as changes to the training data, preprocessing steps, model configuration, or API. When a new model is deployed, I would document the date, model version, training data used, and evaluation results. Previous working versions should be retained when possible so the system can be rolled back if a newer deployment causes unexpected problems.

---