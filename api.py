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
# -----------------------------------------------------------------------
# TODO (Lisa): replace the lecturer's SME revenue model with OUR regressor.
#   1. Copy our saved file into model/ and change the filename below.
#   2. Rename the URL to describe our model:
#      /api/v1/models/<our-model-name>/predictions
#   3. Update the sample request, the response keys and the "model" text.
#   4. Remove the np.expm1(...) step unless our model was ALSO trained on
#      log1p(target).
#   5. Keep get_payload(...) and to_input_row(...) as they are.
# =======================================================================
regression_pipeline = load_artifact("lasso_regressor_for_sme_revenue.joblib")
REGRESSION_FIELDS = expected_fields(regression_pipeline)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Technology", "county": "Nairobi", "quarter": "Q3",
#             "business_age_years": 4.5, "num_employees": 12,
#             "owner_education_level": "Secondary",
#             "monthly_marketing_spend_kes": 45000,
#             "avg_daily_foot_traffic": null,
#             "online_presence_score": 78.0, "loan_amount_kes": 250000,
#             "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
#             "competitor_density": 5, "customer_satisfaction_score": 4.2}'
@app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = get_payload(REGRESSION_FIELDS)
    input_row = to_input_row(payload, REGRESSION_FIELDS)

    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    # Trained on log1p(revenue), so invert that to report KES.
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

    return jsonify({
        "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Lasso Regressor",
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
# TODO (Amiir): replace the lecturer's mall customer model with OUR k-Means
# model.
#   1. Copy our saved file into model/ and change the filename below.
#      If we saved the scaler and k-Means in separate files, load each one.
#   2. Rename the URL: /api/v1/models/<our-model-name>/predictions
#   3. Update the sample request, the response and the "model" text
#      (our k and features). Add cluster names if our lab described them.
#   4. If we did not scale, drop the scaler and call kmeans.predict(...)
#      directly. Keep get_payload(...) and to_input_row(...).
# =======================================================================
clustering_artifact = load_artifact("kmeans_mall_customer_segmentation.joblib")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
CLUSTERING_FIELDS = expected_fields(clustering_scaler)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    payload = get_payload(CLUSTERING_FIELDS)
    input_row = to_input_row(payload, CLUSTERING_FIELDS)

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# =======================================================================
# MEMBER 4 (Felista): ASSOCIATION RULE MINING (recommender)
# -----------------------------------------------------------------------
# TODO (Felista): replace the lecturer's e-commerce rules with OUR rules.
#   1. Copy our saved file into model/ and change the filename below.
#   2. Rename the URL: /api/v1/models/<our-model-name>/predictions
#   3. If our artifact is shaped differently (e.g., an mlxtend rules
#      DataFrame instead of a lookup dict, or no sku_lookup), adapt the
#      matching loop and the product names. Keep the input checks.
#   4. Update the sample requests with items that really appear in our
#      rules, so a recommendation comes back.
# =======================================================================
arm_artifact = load_artifact("apriori_recommendation_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = get_payload(["cart"])

    cart_items = payload["cart"]
    if not isinstance(cart_items, list) or not cart_items:
        raise InputError("'cart' must be a non-empty list, e.g. [\"H06\", \"H07\"].")
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
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "message": "" if recommendations else "No rule matches this cart. Try other items.",
        "model": "Apriori association rules, cleaned + precomputed lookup",
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