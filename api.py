"""
api.py

Our group's Flask API. It serves FOUR machine learning artifacts that were
trained and saved in earlier labs. They are loaded from the model/ folder
once, at startup. NOTHING is ever trained in this file.

    1. Regression       -- MEMBER 1 (Lisa)
    2. Classification   -- MEMBER 2 (Stephanie)
    3. Clustering       -- MEMBER 3 (Amiir)
    4. Association Rule -- MEMBER 4 (Felista)
       Mining (ARM)

STATUS: the four MEMBER blocks below still serve the lecturer's in-class
models, so the API keeps working while we swap them out. Each member replaces
ONLY their own block with our group's model. The SHARED sections (CORS,
helpers, health check, error handling) belong to Mariam: tell her before
changing them.

Run a local development server:

    python api.py        ->  http://127.0.0.1:5000/api/v1/health
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)


# =======================================================================
# SHARED (Member 6- Mariam): CORS (Cross-Origin Resource Sharing)
# =======================================================================
# A browser blocks JavaScript on one origin (e.g., a page served from
# http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000) unless the API allows it. These are the
# origins our frontend pages are served from during development.
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost", "https://127.0.0.1",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # PyCharm's built-in web server
    "null",  # a frontend .html file opened directly (file://)
]}})


# =======================================================================
# SHARED (Mariam): model folder and input helpers
# =======================================================================
# An absolute path, so model/ is found no matter which folder the API is
# started from (PyCharm, a terminal, Docker or Render).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_artifact(filename):
    """Load one saved artifact from model/. Training never happens here."""
    return joblib.load(os.path.join(MODEL_DIR, filename))


def expected_fields(fitted_object):
    """The column names an artifact was fitted on, read from the artifact
    itself (scikit-learn stores them when it is fitted on a DataFrame).
    Returns [] if the artifact does not record them."""
    return [str(name) for name in getattr(fitted_object, "feature_names_in_", [])]


class InputError(Exception):
    """Bad or missing input from the client. Always returned as HTTP 400."""


def get_payload(required_fields=()):
    """Return the JSON body as a dict, or raise InputError if the body is
    missing or empty, or if any required field is absent.

    A field that is PRESENT but null is allowed on purpose: our pipelines'
    imputers handle structural missingness (see the lecturer's notes on
    avg_daily_foot_traffic and collateral_value_kes)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        raise InputError(
            "Send a non-empty JSON object with the header "
            "'Content-Type: application/json'."
        )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise InputError("Missing input field(s): " + ", ".join(missing))
    return payload


def to_input_row(payload, fields):
    """One-row DataFrame holding the fields the model expects, in the
    order it was trained on. Any extra keys in the request are ignored."""
    row = pd.DataFrame([payload])
    return row[fields] if fields else row


# =======================================================================
# SHARED (Member 6 - Mariam): health check
# =======================================================================
# Lists every prediction endpoint automatically, so it stays correct as
# members rename their routes. Render and Docker can also use it to check
# that the API is up.
@app.route("/", methods=["GET"])
@app.route("/api/v1/health", methods=["GET"])
def health():
    endpoints = sorted(
        rule.rule for rule in app.url_map.iter_rules() if "POST" in rule.methods
    )
    return jsonify({
        "status": "ok",
        "message": "All four artifacts are loaded. Send a POST request to an endpoint below.",
        "prediction_endpoints": endpoints,
    })


# =======================================================================
# MEMBER 1 (Lisa): REGRESSION
# =======================================================================
regression_pipeline = load_artifact("student_exam_score_regressor.joblib")
REGRESSION_FIELDS = expected_fields(regression_pipeline)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/student-exam-score-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"major_subject": "Engineering", "parental_education_level": "Bachelor'\''s",
#            "internet_access_quality": "Good", "part_time_job": "No",
#            "study_group_participation": "Yes", "study_hours_per_week": 9.5,
#            "attendance_rate_pct": 88.0, "previous_gpa": 3.4,
#            "sleep_hours_per_night": 7.0, "extracurricular_activities_count": 1,
#            "stress_level_score": 5.0, "tutoring_hours_per_week": 2.0}'
@app.route("/api/v1/models/student-exam-score-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = get_payload(REGRESSION_FIELDS)
    input_row = to_input_row(payload, REGRESSION_FIELDS)

    # TransformedTargetRegressor automatically runs target_transformer.inverse_transform()
    predicted_exam_score = float(regression_pipeline.predict(input_row)[0])

    return jsonify({
        "predicted_final_exam_score": round(predicted_exam_score, 2),
        "model": "Ridge Regressor (with Yeo-Johnson target transformation)",
    })


# =======================================================================
# MEMBER 2 (Stephanie): CLASSIFICATION
# =======================================================================
classification_artifact = load_artifact("fraud_model_bundle.joblib")
classification_pipeline = classification_artifact["model"]
CLASSIFICATION_MODEL_NAME = classification_artifact.get(
    "model_name", "Tuned Logistic Regression"
)
CLASSIFICATION_THRESHOLD = float(classification_artifact["threshold"])
CLASSIFICATION_FIELDS = list(
    classification_artifact.get("features")
    or expected_fields(classification_pipeline)
)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/insurance-fraud-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"policy_type": "Comprehensive", "claim_type": "Collision",
#             "incident_location_type": "Highway", "witnesses_present": "No",
#             "agent_flagged_history": "Yes", "police_report_filed": "No",
#             "claimant_age": 34, "policy_tenure_years": 1.2,
#             "prior_claims_count": 3, "days_to_report_claim": 18,
#             "claim_amount_kes": 480000.0,
#             "claim_amount_to_policy_limit_ratio": 0.92,
#             "customer_credit_score": 540.0}'
@app.route("/api/v1/models/insurance-fraud-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    payload = get_payload(CLASSIFICATION_FIELDS)
    # Convert JSON null (Python None) to np.nan so SimpleImputer handles missingness
    cleaned_payload = {
        key: (np.nan if val is None else val) for key, val in payload.items()
    }
    input_row = to_input_row(cleaned_payload, CLASSIFICATION_FIELDS)

    probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = list(classification_pipeline.classes_)
    fraud_idx = class_labels.index(1) if 1 in class_labels else 1

    fraud_probability = float(probabilities[fraud_idx])
    legit_probability = float(probabilities[1 - fraud_idx])
    is_fraud = fraud_probability >= CLASSIFICATION_THRESHOLD

    return jsonify({
        "predicted_claim_status": "Fraudulent" if is_fraud else "Legitimate",
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(fraud_probability, 4),
        "decision_threshold": CLASSIFICATION_THRESHOLD,
        "class_probabilities": {
            "Legitimate": round(legit_probability, 4),
            "Fraudulent": round(fraud_probability, 4),
        },
        "model": (
            f"{CLASSIFICATION_MODEL_NAME} "
            f"(SMOTE-resampled at training, threshold={CLASSIFICATION_THRESHOLD:.2f})"
        ),
    })


# =======================================================================
# MEMBER 3 (Amiir): CLUSTERING
# -----------------------------------------------------------------------
# Load the bank wealth-management model and its training-time preprocessing.
# =======================================================================
clustering_artifact = load_artifact("kmeans_bank_wealth_segmentation.joblib")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
CLUSTERING_FEATURES = clustering_artifact["feature_order"]
CLUSTER_REGION_MEDIANS = clustering_artifact["region_median_digital_score"]
CLUSTER_NAMES = clustering_artifact["segment_names"]
CLUSTERING_INPUT_FIELDS = [
    "account_balance_kes",
    "transactions_per_month",
    "years_as_customer",
    "digital_banking_usage_score",
]

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/bank-wealth-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"account_balance_kes": 2400000, "transactions_per_month": 9, \
#             "years_as_customer": 11, "digital_banking_usage_score": 55}'
@app.route("/api/v1/models/bank-wealth-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    payload = get_payload(CLUSTERING_INPUT_FIELDS)
    try:
        account_balance = float(payload["account_balance_kes"])
        transactions_per_month = float(payload["transactions_per_month"])
        years_as_customer = float(payload["years_as_customer"])
        digital_value = payload["digital_banking_usage_score"]
        digital_score = None if digital_value is None else float(digital_value)
    except (TypeError, ValueError, OverflowError):
        raise InputError("All clustering inputs must be numbers, except a null digital score.")

    numeric_values = [account_balance, transactions_per_month, years_as_customer]
    if digital_score is not None:
        numeric_values.append(digital_score)
    if not np.isfinite(numeric_values).all():
        raise InputError("Clustering inputs must be finite numbers.")
    if account_balance <= 0:
        raise InputError("account_balance_kes must be greater than zero.")
    if transactions_per_month < 0 or years_as_customer < 0:
        raise InputError("Transactions and customer tenure cannot be negative.")
    if digital_score is not None and not 0 <= digital_score <= 100:
        raise InputError("digital_banking_usage_score must be between 0 and 100.")

    if digital_score is None:
        branch_region = str(payload.get("branch_region", "")).strip()
        if branch_region not in CLUSTER_REGION_MEDIANS:
            raise InputError("A known branch_region is required when the digital score is null.")
        digital_score = float(CLUSTER_REGION_MEDIANS[branch_region])

    transformed_values = {
        "log10_account_balance": float(np.log10(account_balance)),
        "transactions_per_month": transactions_per_month,
        "years_as_customer": years_as_customer,
        "digital_banking_usage_score": digital_score,
    }
    input_row = pd.DataFrame([{
        feature: transformed_values[feature] for feature in CLUSTERING_FEATURES
    }])
    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "model": "kmeans",
        "cluster": assigned_cluster,
        "cluster_name": CLUSTER_NAMES.get(assigned_cluster, f"Cluster {assigned_cluster}"),
    })


# =======================================================================
# MEMBER 4 (Felista): ASSOCIATION RULE MINING (recommender)
# =======================================================================
arm_artifact = load_artifact("arm_pharmacy_basket_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/pharmacy-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["D05", "D04"]}'
@app.route("/api/v1/models/pharmacy-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = get_payload(["cart"])

    cart_items = payload["cart"]
    if not isinstance(cart_items, list) or not cart_items:
        raise InputError("'cart' must be a non-empty list, e.g. [\"D05\", \"D04\"].")
    cart = {str(item).strip() for item in cart_items}

    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        raise InputError("'top_n' must be a whole number, e.g. 3.")
    if top_n < 1:
        raise InputError("'top_n' must be 1 or more.")

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.get(sku, str(sku)),
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "message": "" if recommendations else "No rule matches this cart. Try other items.",
        "model": "Apriori pharmacy basket association rules (precomputed lookup)",
    })


# =======================================================================
# SHARED (Mariam): error handling -- every error is returned as JSON
# =======================================================================
@app.errorhandler(InputError)
def handle_input_error(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(HTTPException)
def handle_http_error(error):
    # 404 (unknown URL), 405 (wrong method, e.g. GET on a POST endpoint), etc.
    return jsonify({"error": error.description, "status": error.code}), error.code


@app.errorhandler(ValueError)
@app.errorhandler(TypeError)
@app.errorhandler(KeyError)
def handle_bad_values(error):
    # Raised by the model when a value has the wrong type or an unknown
    # column/category, e.g. "abc" sent for a numeric field.
    return jsonify({
        "error": "The model could not use this input. Check the field names and value types.",
        "details": str(error),
    }), 400


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unexpected error while handling a request")
    return jsonify({"error": "Internal server error. Check the server logs."}), 500


# =======================================================================
# Entry point (development server only)
# =======================================================================
if __name__ == "__main__":
    # Docker and Render run this app with Gunicorn instead, which imports
    # `app` directly and never runs this block. debug=True gives auto-reload
    # and an interactive debugger -- NEVER use it on a public server.
    app.run(debug=True)