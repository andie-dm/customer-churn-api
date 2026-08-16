import pandas as pd


def prepare_customer_data(customer):
    """
    Convert raw customer JSON into a one-row DataFrame
    that matches the format expected by the transformer.
    """

    customer_df = pd.DataFrame([customer])

    if "customerID" in customer_df.columns:
        customer_df = customer_df.drop(columns=["customerID"])

    if "Unnamed: 0" in customer_df.columns:
        customer_df = customer_df.drop(columns=["Unnamed: 0"])

    if "TotalCharges" in customer_df.columns:
        customer_df["TotalCharges"] = pd.to_numeric(
            customer_df["TotalCharges"],
            errors="coerce"
        )

    return customer_df