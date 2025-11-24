import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Resolve the model path robustly for both local runs and Streamlit Cloud
_here = Path(__file__).resolve().parent
_candidates = [
    _here / 'artifacts' / 'model_data.joblib',            # e.g., repo/app/artifacts/model_data.joblib
    _here.parent / 'artifacts' / 'model_data.joblib',     # e.g., repo/artifacts/model_data.joblib
]

_model_path = next((p for p in _candidates if p.exists()), None)
if _model_path is None:
    raise FileNotFoundError(
        f"Model artifact not found. Looked in: {', '.join(str(p) for p in _candidates)}"
    )

# Load the model and its components with clearer error if versions mismatch
try:
    model_data = joblib.load(_model_path)
except Exception as e:
    raise RuntimeError(
        "Failed to load model artifact. This is commonly due to a version mismatch between\n"
        "the environment (numpy/scikit-learn/joblib) and the versions used to create the artifact.\n"
        f"Artifact path: {_model_path}\nOriginal error: {e}"
    ) from e

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                    loan_purpose, loan_type):
    # Create a dictionary with input values and dummy values for missing features
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # additional dummy fields just for scaling purpose
        'number_of_dependants': 1,  # Dummy value
        'years_at_current_address': 1,  # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }

    # Ensure all columns for features and cols_to_scale are present
    df = pd.DataFrame([input_data])

    # Ensure only required columns for scaling are scaled
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure the DataFrame contains only the features expected by the model
    df = df[features]

    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    # Prepare input data
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating