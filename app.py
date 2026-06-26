import os
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Optional

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import joblib
import smtplib
from email.message import EmailMessage
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    # If python-dotenv is not installed, environment variables should be set externally.
    pass

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATABASE_URL = (
    "postgresql+psycopg2://postgres:1234@localhost/FRAUDE_DETECTION"
)
MODEL_PATH = "xgboost_model.pkl"
TABLE_NAME = "bank_transactions"
FRAUD_THRESHOLD = 0.40
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "oflamoha2005@gmail.com")
EMAIL_SENDER = os.getenv("ALERT_EMAIL_SENDER", "alerts@example.com")
SMTP_SERVER = os.getenv("ALERT_SMTP_SERVER", "localhost")
SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", 25))
SMTP_USERNAME = os.getenv("ALERT_SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("ALERT_SMTP_PASSWORD")
LOG_FILE = "app.log"
MODEL_FEATURES = [
    "merchant_name",
    "transaction_category",
    "transaction_amount",
    "customer_city",
    "customer_job",
    "hour",
    "day_of_week",
]
RAW_SCHEMA_COLUMNS = [
    "transaction_id",
    "credit_card_number",
    "merchant_name",
    "transaction_category",
    "transaction_amount",
    "customer_city",
    "customer_job",
    "transaction_datetime",
    "fraud_result",
]

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------

def setup_logger(name: str = "fraud_app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# -----------------------------------------------------------------------------
# Database Utilities
# -----------------------------------------------------------------------------

def get_engine() -> Engine:
    logger.debug("Creating database engine.")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return engine


def initialize_database(engine: Engine) -> None:
    logger.debug("Initializing the PostgreSQL fraud transaction table.")
    create_table_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            transaction_id VARCHAR(100) PRIMARY KEY,
            credit_card_number VARCHAR(50),
            merchant_name VARCHAR(255),
            transaction_category VARCHAR(100),
            transaction_amount NUMERIC(10, 2),
            customer_city VARCHAR(150),
            customer_job VARCHAR(150),
            transaction_datetime TIMESTAMP WITHOUT TIME ZONE,
            fraud_result VARCHAR(10)
        )
        """
    )
    with engine.connect() as connection:
        connection.execute(create_table_sql)
        connection.commit()
    logger.info("Database table ensured: %s", TABLE_NAME)


def fetch_existing_transaction_keys(engine: Engine) -> set:
    query = text(f"SELECT transaction_id FROM {TABLE_NAME}")
    with engine.connect() as connection:
        result = connection.execute(query)
        existing_keys = {row[0] for row in result.fetchall()}
    logger.debug("Fetched %d existing transaction keys.", len(existing_keys))
    return existing_keys


def upsert_fraud_results(engine: Engine, updates: pd.DataFrame) -> None:
    logger.debug("Updating fraud result values in the database.")
    # Perform a single bulk UPDATE using a VALUES list joined as a subquery.
    records = updates[["transaction_id", "fraud_result"]].dropna(subset=["transaction_id"]) \
        .astype(str)
    if records.empty:
        logger.debug("No fraud result records to update.")
        return

    values = records.to_dict(orient="records")
    # Build named parameter placeholders for safety
    params = {}
    placeholder_items = []
    for i, rec in enumerate(values):
        t_key = f"t{i}"
        r_key = f"r{i}"
        params[t_key] = rec["transaction_id"]
        params[r_key] = rec["fraud_result"]
        placeholder_items.append(f"(:{t_key}, :{r_key})")

    values_sql = ",".join(placeholder_items)
    update_sql = text(
        f"UPDATE {TABLE_NAME} AS bt SET fraud_result = v.fraud_result "
        f"FROM (VALUES {values_sql}) AS v(transaction_id, fraud_result) "
        f"WHERE bt.transaction_id = v.transaction_id"
    )

    with engine.connect() as connection:
        connection.execute(update_sql, params)
        connection.commit()
    logger.info("Bulk-updated fraud_result for %d records.", len(values))

# -----------------------------------------------------------------------------
# Data Wrangling and Preprocessing
# -----------------------------------------------------------------------------

def rename_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Renaming raw dataset columns to match PostgreSQL schema.")
    rename_map = {
        "trans_date_trans_time": "transaction_datetime",
        "cc_num": "credit_card_number",
        "merchant": "merchant_name",
        "category": "transaction_category",
        "amt": "transaction_amount",
        "first": "customer_first_name",
        "last": "customer_last_name",
        "gender": "customer_gender",
        "street": "customer_street",
        "city": "customer_city",
        "state": "customer_state",
        "zip": "zip_code",
        "lat": "customer_latitude",
        "long": "customer_longitude",
        "city_pop": "city_population",
        "job": "customer_job",
        "dob": "customer_birth_date",
        "trans_num": "transaction_id",
        "transaction_id": "transaction_id",
        "unix_time": "unix_timestamp",
        "merch_lat": "merchant_latitude",
        "merch_long": "merchant_longitude",
        "is_fraud": "fraud_label",
    }
    return df.rename(columns=rename_map)


def prepare_raw_database_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing raw dataset for immediate PostgreSQL persistence.")
    raw = rename_raw_columns(raw_df.copy())

    missing_columns = set(RAW_SCHEMA_COLUMNS) - set(raw.columns)
    if missing_columns:
        logger.warning(
            "Uploaded dataset is missing expected columns: %s",
            sorted(missing_columns),
        )

    raw = raw.assign(
        fraud_result=np.nan,
    )

    raw = raw.loc[:, raw.columns.intersection(RAW_SCHEMA_COLUMNS)]

    for column in ["transaction_amount"]:
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    if "transaction_datetime" in raw:
        raw["transaction_datetime"] = pd.to_datetime(
            raw["transaction_datetime"], errors="coerce"
        )

    raw["transaction_datetime"] = raw["transaction_datetime"].where(
        raw["transaction_datetime"].notna(), None
    )

    raw = raw.drop_duplicates(subset=["transaction_id"], keep="first")
    logger.info("Prepared raw dataset with %d records after deduplication.", len(raw))
    return raw


def prepare_features_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating a dedicated feature engineering copy for prediction.")
    features = raw_df.copy()

    if "transaction_datetime" in features:
        features["transaction_datetime"] = pd.to_datetime(
            features["transaction_datetime"], errors="coerce"
        )
    else:
        raise ValueError("transaction_datetime column is required for feature extraction.")

    # Ensure transaction amount is numeric before model consumption.
    if "transaction_amount" in features:
        features["transaction_amount"] = pd.to_numeric(
            features["transaction_amount"], errors="coerce"
        ).fillna(0.0).astype(float)

    features["hour"] = features["transaction_datetime"].dt.hour.fillna(0).astype(int)
    features["day_of_week"] = (
        features["transaction_datetime"].dt.dayofweek.fillna(0).astype(int)
    )

    for feature_name in [
        "merchant_name",
        "customer_job",
        "customer_city",
        "transaction_category",
    ]:
        if feature_name not in features.columns:
            raise ValueError(f"Feature column missing: {feature_name}")

    for feature_name in ["merchant_name", "customer_job", "customer_city"]:
        frequency_map = features[feature_name].value_counts(dropna=False)
        logger.debug("Mapping frequency for %s with %d unique values.", feature_name, len(frequency_map))
        features[feature_name] = features[feature_name].map(frequency_map).fillna(0)

    transaction_category_freq = features["transaction_category"].value_counts(dropna=False)
    features["transaction_category"] = features["transaction_category"].map(
        transaction_category_freq
    ).fillna(0)
    logger.debug("Mapped transaction_category frequency.")

    model_features = features.loc[:, MODEL_FEATURES].copy()
    # Convert all model features to numeric values and fill missing values.
    for feature_name in MODEL_FEATURES:
        if model_features[feature_name].dtype == object:
            model_features[feature_name] = pd.to_numeric(
                model_features[feature_name], errors="coerce"
            )
    model_features = model_features.fillna(0.0).astype(float)

    return model_features

    return model_features

# -----------------------------------------------------------------------------
# Prediction and Alerting
# -----------------------------------------------------------------------------

def load_model(path: str) -> object:
    logger.info("Loading pre-trained model from %s.", path)
    model = joblib.load(path)
    logger.info("Model loaded successfully.")
    return model


def get_positive_probability(model: object, features: pd.DataFrame) -> np.ndarray:
    logger.debug("Calculating fraud probability values.")
    probabilities = model.predict_proba(features)
    class_index = 1
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            class_index = classes.index(1)
    return probabilities[:, class_index]


def create_prediction_summary(model: object, features: pd.DataFrame) -> pd.DataFrame:
    fraud_probabilities = get_positive_probability(model, features)
    result_series = np.where(
        fraud_probabilities >= FRAUD_THRESHOLD, "Fraud", "Legit"
    )
    logger.info("Predicted %d fraud records out of %d.",
                int((result_series == "Fraud").sum()), len(result_series))
    predictions = pd.DataFrame(
        {
            "fraud_probability": fraud_probabilities,
            "fraud_result": result_series,
        },
        index=features.index,
    )
    return predictions


def mask_credit_card_number(card_number: Optional[str]) -> str:
    if not isinstance(card_number, str) or not card_number.strip():
        return "UNKNOWN"
    digits = [ch for ch in card_number if ch.isdigit()]
    if len(digits) <= 4:
        return "*" * max(0, len(digits))
    masked = "*" * (len(digits) - 4) + "".join(digits[-4:])
    return masked


def build_alert_email_body(fraud_df: pd.DataFrame, masked_cards: List[str]) -> str:
    lines = [
        "Dear Security Team,",
        "",
        "The fraud detection system has identified suspicious activity that requires immediate attention.",
        "Please review the details below and take appropriate action for the impacted accounts.",
        "",
        f"Total Fraudulent Transactions Detected: {len(fraud_df)}",
        "",
        "Incident Summary:",
        "",
    ]

    for _, row in fraud_df.iterrows():
        lines.append(
            f"- Transaction ID: {row['transaction_id']}\n"
            f"  Card Number: {row['masked_credit_card']}\n"
            f"  Merchant: {row['merchant_name']}\n"
            f"  Category: {row['transaction_category']}\n"
            f"  Amount: ${row['transaction_amount']:.2f}\n"
            f"  Customer City: {row['customer_city']}\n"
            f"  Customer Job: {row['customer_job']}\n"
            f"  Transaction Date: {row['transaction_datetime']}\n"
            f"  Fraud Probability: {row['fraud_probability']:.4f}\n"
        )

    lines.append("")
    lines.append("Affected masked card numbers:")
    lines.extend([f"- {card}" for card in masked_cards])
    lines.append("")
    lines.append("Please escalate this incident to the appropriate fraud response team immediately.")
    lines.append("If you require additional transaction details, consult the fraud dashboard or database records.")
    lines.append("")
    lines.append("Regards,")
    lines.append("Fraud Detection System")
    return "\n".join(lines)


def send_alert_email(subject: str, body: str, recipient: str) -> bool:
    logger.info("Dispatching alert email to %s.", recipient)
    logger.debug(
        "SMTP config: server=%s port=%d username=%s sender=%s recipient=%s password_present=%s",
        SMTP_SERVER,
        SMTP_PORT,
        SMTP_USERNAME,
        EMAIL_SENDER,
        recipient,
        bool(SMTP_PASSWORD),
    )
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = EMAIL_SENDER
    message["To"] = recipient
    message.set_content(body)

    try:
        if SMTP_USERNAME and SMTP_PASSWORD:
            logger.debug("Connecting to SMTP with credentials.")
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
                smtp.send_message(message)
        else:
            logger.debug("Connecting to local SMTP server at %s:%d.", SMTP_SERVER, SMTP_PORT)
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as smtp:
                smtp.send_message(message)
        logger.info("Alert email successfully sent.")
        return True
    except Exception as exc:
        logger.error("Failed to send fraud alert email: %s", exc, exc_info=True)
        return False

# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Credit Card Fraud Detection Dashboard")
    st.markdown(
        "Upload a transaction CSV file to persist raw records, score them with the pre-trained XGBoost model, "
        "and receive a consolidated fraud alert when suspicious activity is found."
    )

    uploaded_file = st.file_uploader(
        "Upload a transaction dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file containing the transaction dataset.")
        return

    try:
        filename = getattr(uploaded_file, "name", "").lower()
        if filename.endswith((".xlsx", ".xls")):
            raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            # default to CSV
            raw_df = pd.read_csv(uploaded_file)

        logger.info(
            "Uploaded dataset contains %d rows and %d columns.", raw_df.shape[0], raw_df.shape[1]
        )
    except Exception as exc:
        logger.error("Error reading uploaded file: %s", exc, exc_info=True)
        st.error("Unable to read the uploaded file. Please verify the file format (CSV or Excel).")
        return

    engine = get_engine()
    initialize_database(engine)

    # ---------------------------
    # 1) Immediate Raw Ingestion
    # ---------------------------
    raw_db_df = prepare_raw_database_df(raw_df)

    # Keep track of all uploaded transaction keys (even if some already exist)
    uploaded_trans = raw_db_df["transaction_id"].dropna().astype(str).unique().tolist()

    existing_keys = fetch_existing_transaction_keys(engine)
    new_rows = raw_db_df[~raw_db_df["transaction_id"].isin(existing_keys)]

    if not new_rows.empty:
        try:
            new_rows.to_sql(
                TABLE_NAME,
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
            )
            logger.info("Immediately inserted %d new raw rows to the DB.", len(new_rows))
        except Exception as exc:
            logger.error("Failed to persist raw rows to the database: %s", exc, exc_info=True)
            st.error("There was an issue saving raw transactions to PostgreSQL.")
            return
    else:
        logger.info("No new raw rows to insert; all transaction_id values already exist in DB.")

    # ------------------------------------------------------------------
    # 2) Feature Engineering & Prediction (operate on DB-stored snapshot)
    # ------------------------------------------------------------------
    if not uploaded_trans:
        st.warning("Uploaded file does not contain any valid 'transaction_id' values.")
        return

    # Read back the exact rows we care about from the database to ensure
    # feature engineering is performed on the persisted, timestamped records.
    placeholders = ", ".join([f":t{i}" for i in range(len(uploaded_trans))])
    params = {f"t{i}": uploaded_trans[i] for i in range(len(uploaded_trans))}
    select_sql = text(
        f"SELECT transaction_id, credit_card_number, merchant_name, transaction_category, "
        f"transaction_amount, customer_city, customer_job, transaction_datetime, fraud_result "
        f"FROM {TABLE_NAME} WHERE transaction_id IN ({placeholders})"
    )

    with engine.connect() as connection:
        result = connection.execute(select_sql, params)
        rows = result.fetchall()
        if not rows:
            st.warning("No persisted rows found in the database for the uploaded transactions.")
            return
        fetched_df = pd.DataFrame(rows, columns=result.keys())

    features_df = prepare_features_for_model(fetched_df)
    model = load_model(MODEL_PATH)
    prediction_df = create_prediction_summary(model, features_df)

    # Align predictions with the original persisted transaction IDs.
    prediction_df = prediction_df.reset_index(drop=True)
    prediction_df["transaction_id"] = fetched_df["transaction_id"].reset_index(drop=True)

    # Bulk update only the transaction_id and fraud_result values.
    try:
        upsert_df = prediction_df[["transaction_id", "fraud_result"]]
        upsert_fraud_results(engine, upsert_df)
    except Exception as exc:
        logger.error("Failed to bulk-update fraud_result in DB: %s", exc, exc_info=True)
        st.error("There was an issue updating fraud results in PostgreSQL.")
        return

    fetched_df["fraud_probability"] = prediction_df["fraud_probability"].values
    fetched_df["fraud_result"] = prediction_df["fraud_result"].values
    fetched_df["masked_credit_card"] = fetched_df["credit_card_number"].apply(mask_credit_card_number)

    total_count = len(fetched_df)
    fraud_count = int((fetched_df["fraud_result"] == "Fraud").sum())
    legit_count = int((fetched_df["fraud_result"] == "Legit").sum())

    st.metric("Total Transactions", total_count)
    st.metric("Fraudulent Transactions", fraud_count)
    st.metric("Legitimate Transactions", legit_count)

    fraud_rows = fetched_df[fetched_df["fraud_result"] == "Fraud"].copy()
    fraud_rows["fraud_probability"] = fraud_rows["fraud_probability"].round(4)
    fraud_rows_display = fraud_rows[
        [
            "transaction_id",
            "masked_credit_card",
            "merchant_name",
            "transaction_category",
            "transaction_amount",
            "customer_city",
            "customer_job",
            "transaction_datetime",
            "fraud_probability",
        ]
    ]

    if not fraud_rows.empty:
        masked_cards = sorted(fraud_rows["masked_credit_card"].unique())
        body = build_alert_email_body(fraud_rows, masked_cards)
        email_subject = (
            f"Urgent Fraud Alert: {fraud_count} Suspicious Transaction(s) Detected"
        )
        email_sent = send_alert_email(email_subject, body, ALERT_EMAIL)

        if email_sent:
            st.success(
                "The transactions and fraud scores were stored successfully. "
                "A professional alert email has been sent to the configured recipient."
            )
        else:
            st.warning(
                "The transactions were stored, but the alert email could not be sent. "
                "Please verify the SMTP configuration and retry."
            )

        st.subheader("Flagged Fraud Transactions")
        st.dataframe(fraud_rows_display.reset_index(drop=True))
    else:
        st.success(
            "The transactions along with their fraud probabilities have been successfully stored in the PostgreSQL database. "
            "No fraud alerts were triggered for this upload."
        )

    logger.info(
        "Processing complete: total=%d, fraud=%d, legit=%d.",
        total_count,
        fraud_count,
        legit_count,
    )


if __name__ == "__main__":
    main()
