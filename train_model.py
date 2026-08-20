import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------------
# 1. LOAD THE GOLD DATASET
# -----------------------------------

df = pd.read_csv("gold_churn_data.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# -----------------------------------
# 2. REMOVE UNNEEDED INDEX COLUMN
# -----------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# -----------------------------------
# 3. CREATE FEATURES X AND TARGET y
# -----------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]


# -----------------------------------
# 4. REMOVE CUSTOMER ID
# -----------------------------------

X = X.drop(columns=["customerID"])


# -----------------------------------
# 5. MAKE TOTALCHARGES NUMERIC
# -----------------------------------

X["TotalCharges"] = pd.to_numeric(
    X["TotalCharges"],
    errors="coerce"
)


# -----------------------------------
# 6. CONVERT TARGET TO 0 AND 1
# -----------------------------------

y = y.map({
    "Yes": 1,
    "No": 0
})


# -----------------------------------
# 7. IDENTIFY COLUMN TYPES
# -----------------------------------

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


# -----------------------------------
# 8. CREATE PREPROCESSOR
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="mean"),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_cols
        )
    ]
)


# -----------------------------------
# 9. TRANSFORM THE DATA
# -----------------------------------

X_cleaned = preprocessor.fit_transform(X)


#10. TRAIN TEST SPLIT
#X_train, X_test, y_train, y_test = train_test_split(
 #   X_cleaned, 
 #   y, test_size=0.2, random_state=42)


# -----------------------------------
# 10. TRAIN MODEL - LOGISTIC REGRESSION
# -----------------------------------

model = LogisticRegression(
    max_iter=5000
)

model.fit(X_cleaned, y)

# 11. ACCURACY

y_pred = model.predict(X_cleaned)

print("Accuracy:",accuracy_score(y, y_pred)) 




# -----------------------------------
# 12. SAVE TRANSFORMER AND MODEL
# -----------------------------------

joblib.dump(
    preprocessor,
    "app/transformer.pkl"
)

joblib.dump(
    model,
    "app/model.pkl"
)


print("Training complete.")
print("Saved app/transformer.pkl")
print("Saved app/model.pkl")