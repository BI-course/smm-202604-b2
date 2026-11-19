import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Load the same saved artifacts as the API. Hugging Face receives a model/
# copy; local runs can use the repository's root model/ folder.
# -----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "model"))

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_bank_wealth_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
CLUSTERING_FEATURES = clustering_artifact["feature_order"]
CLUSTER_REGION_MEDIANS = clustering_artifact["region_median_digital_score"]
CLUSTER_NAMES = clustering_artifact["segment_names"]

def load_artifact(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


regression_pipeline = load_artifact("student_exam_score_regressor.joblib")
fraud_artifact = load_artifact("fraud_model_bundle.joblib")
classification_pipeline = fraud_artifact["model"]
classification_threshold = float(fraud_artifact["threshold"])
arm_artifact = load_artifact("arm_pharmacy_basket_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]
REGRESSION_FIELDS = list(regression_pipeline.feature_names_in_)
CLASSIFICATION_FIELDS = list(fraud_artifact["features"])

REGRESSION_DEFAULTS = {
    "major_subject": "Engineering", "parental_education_level": "Bachelor's",
    "internet_access_quality": "Good", "part_time_job": "No",
    "study_group_participation": "Yes", "study_hours_per_week": 9.5,
    "attendance_rate_pct": 88.0, "previous_gpa": 3.4,
    "sleep_hours_per_night": 7.0, "extracurricular_activities_count": 1,
    "stress_level_score": 5.0, "tutoring_hours_per_week": 2.0,
}
CLASSIFICATION_DEFAULTS = {
    "policy_type": "Comprehensive", "claim_type": "Collision",
    "incident_location_type": "Highway", "witnesses_present": "No",
    "agent_flagged_history": "Yes", "police_report_filed": "No",
    "claimant_age": 34, "policy_tenure_years": 1.2,
    "prior_claims_count": 3, "days_to_report_claim": 18,
    "claim_amount_kes": 480000.0,
    "claim_amount_to_policy_limit_ratio": 0.92, "customer_credit_score": 540.0,
}


def make_inputs(fields, defaults):
    return [
        gr.Textbox(label=field, value=defaults[field])
        if isinstance(defaults[field], str)
        else gr.Number(label=field, value=defaults[field])
        for field in fields
    ]


# -----------------------------------------------------------------------
# 1. Regression -- predict a student's final exam score
# -----------------------------------------------------------------------
def predict_exam_score(*values):
    row = pd.DataFrame([dict(zip(REGRESSION_FIELDS, values))])
    prediction = float(regression_pipeline.predict(row)[0])
    return {
        "model": "student_exam_score_regressor",
        "predicted_final_exam_score": round(prediction, 2),
    }


# -----------------------------------------------------------------------
# 2. Classification -- predict insurance claim fraud
# -----------------------------------------------------------------------
def predict_fraud(*values):
    row = pd.DataFrame([dict(zip(CLASSIFICATION_FIELDS, values))])
    probabilities = classification_pipeline.predict_proba(row)[0]
    class_labels = list(classification_pipeline.classes_)
    fraud_index = class_labels.index(1) if 1 in class_labels else 1
    fraud_probability = float(probabilities[fraud_index])
    legit_probability = float(probabilities[1 - fraud_index])
    is_fraud = fraud_probability >= classification_threshold
    return {
        "model": fraud_artifact.get("model_name", "Insurance fraud classifier"),
        "predicted_claim_status": "Fraudulent" if is_fraud else "Legitimate",
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 4),
        "decision_threshold": classification_threshold,
        "class_probabilities": {
            "Legitimate": round(legit_probability, 4),
            "Fraudulent": round(fraud_probability, 4),
        },
    }


# -----------------------------------------------------------------------
# 3. Clustering -- assign a bank account to a wealth-management segment
# -----------------------------------------------------------------------
def predict_segment(account_balance_kes, transactions_per_month, years_as_customer,
                    digital_banking_usage_score, branch_region):
    try:
        account_balance = float(account_balance_kes)
        transactions = float(transactions_per_month)
        tenure = float(years_as_customer)
        digital_score = (
            None if digital_banking_usage_score is None
            else float(digital_banking_usage_score)
        )
    except (TypeError, ValueError, OverflowError):
        raise gr.Error("Enter numeric values; the digital score may be left blank.")

    values = [account_balance, transactions, tenure]
    if digital_score is not None:
        values.append(digital_score)
    if not np.isfinite(values).all():
        raise gr.Error("Inputs must be finite numbers.")
    if account_balance <= 0:
        raise gr.Error("Account balance must be greater than zero.")
    if transactions < 0 or tenure < 0:
        raise gr.Error("Transactions and customer tenure cannot be negative.")
    if digital_score is not None and not 0 <= digital_score <= 100:
        raise gr.Error("Digital usage score must be between 0 and 100.")
    if digital_score is None:
        if branch_region not in CLUSTER_REGION_MEDIANS:
            raise gr.Error("Choose a known branch region when the digital score is blank.")
        digital_score = float(CLUSTER_REGION_MEDIANS[branch_region])

    transformed_values = {
        "log10_account_balance": float(np.log10(account_balance)),
        "transactions_per_month": transactions,
        "years_as_customer": tenure,
        "digital_banking_usage_score": digital_score,
    }
    input_row = pd.DataFrame([{
        feature: transformed_values[feature] for feature in CLUSTERING_FEATURES
    }])
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    cluster_name = CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")
    return f"{cluster_name} (cluster {cluster})"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend products for a shopping cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
    cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
    if not cart:
        raise gr.Error("Enter at least one SKU, separated by commas.")
    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
    return {
        "model": "pharmacy association rules",
        "cart": sorted(cart),
        "recommendations": [
            {
                "sku": str(sku),
                "product_name": sku_lookup.get(sku, str(sku)),
                "confidence": round(float(confidence), 3),
                "lift": round(float(lift), 2),
            }
            for sku, (confidence, lift) in ranked[:int(top_n)]
        ],
        "message": "" if ranked else "No rule matches this cart. Try other items.",
    }


# -----------------------------------------------------------------------
# One Gradio app with one tab per model.
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Demos") as demo:
    gr.Markdown("# BBT 4206 Model Demos")

    with gr.Tab("Exam score"):
        gr.Interface(
            fn=predict_exam_score,
            inputs=make_inputs(REGRESSION_FIELDS, REGRESSION_DEFAULTS),
            outputs=gr.JSON(label="Prediction"),
        )

    with gr.Tab("Fraud classification"):
        gr.Interface(
            fn=predict_fraud,
            inputs=make_inputs(CLASSIFICATION_FIELDS, CLASSIFICATION_DEFAULTS),
            outputs=gr.JSON(label="Prediction"),
        )

    with gr.Tab("Bank Wealth Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Account Balance (KES)", value=2400000),
                gr.Number(label="Transactions per Month", value=9),
                gr.Number(label="Years as Customer", value=11),
                gr.Number(label="Digital Banking Usage Score (leave blank if unknown)", value=55),
                gr.Dropdown(
                    label="Branch Region",
                    choices=sorted(CLUSTER_REGION_MEDIANS),
                    value="Nairobi",
                ),
            ],
            outputs=gr.Textbox(label="Assigned Segment"),
        )

    with gr.Tab("Pharmacy recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Basket SKUs", value="D05, D04"),
                gr.Slider(1, 10, value=3, step=1, label="Recommendations"),
            ],
            outputs=gr.JSON(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()