#!/usr/bin/env bash
# Smoke test for our ML API (Member 5, Ian).
# Usage: bash tests/smoke_test.sh https://localhost
#    or: bash tests/smoke_test.sh http://127.0.0.1:5000
#    or: bash tests/smoke_test.sh https://<our-service>.onrender.com
#
# When a member swaps in a new model, update their _URL and _BODY below
# from the sample cURL in their MEMBER block of api.py.
# Sample inputs copied from each MEMBER block in api.py (19 Nov).

BASE="${1:-https://localhost}"
PASS=0
FAIL=0

# check <label> <expected HTTP code> <method> <path> [json body]
check() {
  local label="$1" expected="$2" method="$3" path="$4" body="$5"
  echo "== $label"
  if [ -n "$body" ]; then
    response=$(curl -sk -w '\n%{http_code}' -X "$method" "$BASE$path" \
      -H "Content-Type: application/json" -d "$body")
  else
    response=$(curl -sk -w '\n%{http_code}' -X "$method" "$BASE$path")
  fi
  code=$(echo "$response" | tail -n1)
  echo "$response" | sed '$d'
  if [ "$code" = "$expected" ]; then
    echo "-> HTTP $code (OK)"; PASS=$((PASS + 1))
  else
    echo "-> HTTP $code (expected $expected)"; FAIL=$((FAIL + 1))
  fi
  echo
}

# ---- Our four models: URL + one sample request each ----
REGRESSION_URL="/api/v1/models/student-exam-score-regressor/predictions"            # Lisa
REGRESSION_BODY='{"major_subject": "Engineering", "parental_education_level": "Bachelor'\''s", "internet_access_quality": "Good", "part_time_job": "No", "study_group_participation": "Yes", "study_hours_per_week": 9.5, "attendance_rate_pct": 88.0, "previous_gpa": 3.4, "sleep_hours_per_night": 7.0, "extracurricular_activities_count": 1, "stress_level_score": 5.0, "tutoring_hours_per_week": 2.0}'
CLASSIFICATION_URL="/api/v1/models/insurance-fraud-classifier/predictions"  # Stephanie
CLASSIFICATION_BODY='{"policy_type": "Comprehensive", "claim_type": "Collision", "incident_location_type": "Highway", "witnesses_present": "No", "agent_flagged_history": "Yes", "police_report_filed": "No", "claimant_age": 34, "policy_tenure_years": 1.2, "prior_claims_count": 3, "days_to_report_claim": 18, "claim_amount_kes": 480000.0, "claim_amount_to_policy_limit_ratio": 0.92, "customer_credit_score": 540.0}'
CLUSTERING_URL="/api/v1/models/bank-wealth-segmenter/predictions"          # Amiir
CLUSTERING_BODY='{"account_balance_kes": 2400000, "transactions_per_month": 9, "years_as_customer": 11, "digital_banking_usage_score": 55}'
RECOMMENDER_URL="/api/v1/models/pharmacy-recommender/predictions"            # Felista
RECOMMENDER_BODY='{"cart": ["D05", "D04"]}'
# ---------------------------------------------

check "Health check"                  200 GET  "/api/v1/health"
check "Regression (Lisa)"             200 POST "$REGRESSION_URL"     "$REGRESSION_BODY"
check "Classification (Stephanie)"    200 POST "$CLASSIFICATION_URL" "$CLASSIFICATION_BODY"
check "Clustering (Amiir)"            200 POST "$CLUSTERING_URL"     "$CLUSTERING_BODY"
check "Recommender (Felista)"         200 POST "$RECOMMENDER_URL"    "$RECOMMENDER_BODY"
check "Missing input (should be 400)" 400 POST "$REGRESSION_URL"     '{"major_subject": "Engineering"}'
check "Empty cart (should be 400)"    400 POST "$RECOMMENDER_URL"    '{"cart": []}'
check "Wrong method (should be 405)"  405 GET  "$REGRESSION_URL"
check "Unknown URL (should be 404)"   404 GET  "/api/v1/does-not-exist"

echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
