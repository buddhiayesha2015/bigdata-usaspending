"""
BigData & Tools USASpending Application
============================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

This module implements a BigData application for interacting with USASpending data.
It provides various endpoints for data visualization, machine learning inference,
and data management using big data tools such as Apache Spark and Cassandra.

**Features:**
- **Dashboard:** Displays award amounts by recipient and sub-agency.
- **Machine Learning Models:** Implements regression, classification, and clustering models for award data.
- **Data Fetching & Insertion:** Retrieves data from the USASpending API and inserts it into Cassandra.
- **Correlation Visualization:** Visualizes correlations within the data.
- **Download History Tracking:** Keeps a log of data downloads.

**Dependencies:**
- **Apache Spark:** 3.4.4
- **Apache Cassandra:** 4.1.7
- **Scala:** 2.12.17
- **Python:** 3.12.8
- **pytest:** 8.3.4
- **OpenJDK:** 11.0.25
- **Flask:** 3.1.0

**Operating System:** Ubuntu 22.04.5 LTS

**Author:** Buddhi Ayesha
**Date:** 2024-12-20
"""

import os
import random
import time
import uuid
from datetime import datetime

import requests
from cassandra.cluster import Cluster
from flask import Flask, render_template, request, redirect, url_for, flash
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from app import config
from app.logging_config import setup_logging

logger = setup_logging()

app = Flask(__name__, template_folder='../templates', static_folder="../static")
logger.info("BigData & Tools USASpending Flask app initialized")

# Get the secret key from the environment variable
app.secret_key = os.getenv("BDT_SECRET_KEY")

spark = SparkSession.builder \
    .appName("FlaskSparkInference") \
    .master("local[*]") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.3.0") \
    .config("spark.cassandra.connection.host", config.CASSANDRA_HOST) \
    .config("spark.cassandra.connection.port", config.CASSANDRA_PORT) \
    .getOrCreate()

# Load Pipeline Models
REG_PIPELINE_PATH = "outputs/pipeline_regression"
CLASS_PIPELINE_PATH = "outputs/pipeline_classification"
CLUSTER_PIPELINE_PATH = "outputs/pipeline_clustering"

pipeline_model_reg = PipelineModel.load(REG_PIPELINE_PATH)
pipeline_model_class = PipelineModel.load(CLASS_PIPELINE_PATH)
pipeline_model_cluster = PipelineModel.load(CLUSTER_PIPELINE_PATH)

try:
    cassandra_cluster = Cluster(contact_points=[config.CASSANDRA_HOST], port=config.CASSANDRA_PORT)
    cassandra_session = cassandra_cluster.connect("usaspending_data")
except Exception as e:
    logger.error(f"Error connecting to Cassandra: {e}")
    spark.stop()
    raise e


# Flask Routes
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Render the dashboard page with various data visualizations.

    This route reads data from Cassandra tables related to award amounts by recipient and
    sub-agency, as well as data for Sankey diagrams and monthly awards. The data is
    processed using Spark and passed to the dashboard template for rendering.

    Returns:
        A rendered HTML template for the dashboard.
    """
    # Read data for the Map
    df_map_recipient = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="total_award_amount_by_recipient", keyspace="usaspending_data")
        .load()
        .orderBy(col("total_award_amount").desc())
        # .limit(20)
    )
    map_recipient_data = [row.asDict() for row in df_map_recipient.collect()]

    df_map_subagency = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="total_award_amount_by_awarding_sub_agency", keyspace="usaspending_data")
        .load()
        .orderBy(col("total_award_amount").desc())
        # .limit(20)
    )
    map_subagency_data = [row.asDict() for row in df_map_subagency.collect()]

    # Read data for Sankey - Top 15 by total_award_amount
    df_sankey = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awarding_sub_agency_by_recipient", keyspace="usaspending_data")
        .load()
        .orderBy(col("total_award_amount").desc())
        .limit(15)
    )
    sankey_data = [row.asDict() for row in df_sankey.collect()]

    # Read data for Pie and Line - Top 30 by total_award_amount
    df_month = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awarding_sub_agency_by_month", keyspace="usaspending_data")
        .load()
        .orderBy(col("total_award_amount").desc())
        .limit(30)
    )
    month_data = [row.asDict() for row in df_month.collect()]

    return render_template(
        "dashboard.html",
        map_recipient_data=map_recipient_data,
        map_subagency_data=map_subagency_data,
        sankey_data=sankey_data,
        month_data=month_data
    )


@app.route("/correlation")
def show_correlation():
    """
    Render the correlation visualization page.

    Returns:
        A rendered HTML template for correlation visualization.
    """
    return render_template("correlation.html")


@app.route("/regression", methods=["GET", "POST"])
def regression():
    """
    Handle regression model inference for award amount prediction.

    On GET request:
        Renders the regression form with default values.

    On POST request:
        Processes form data, performs regression prediction using the loaded pipeline model,
        and flashes the prediction result.

    Returns:
        A rendered HTML template for regression or a redirect with flashed messages.
    """
    awarding_agency = "Department of Defense"
    awarding_sub_agency = None
    contract_award_type = None
    funding_agency = None
    funding_sub_agency = None
    month = 1
    year = 2023
    award_amount = 0.0

    if request.method == "POST":
        awarding_agency = request.form.get("awarding_agency", "")
        awarding_sub_agency = request.form.get("awarding_sub_agency", "")
        contract_award_type = request.form.get("contract_award_type", "")
        funding_agency = request.form.get("funding_agency", "")
        funding_sub_agency = request.form.get("funding_sub_agency", "")
        month = int(request.form.get("month", 1))
        year = int(request.form.get("year", 2023))

        data = [(
            awarding_agency,
            awarding_sub_agency,
            contract_award_type,
            funding_agency,
            funding_sub_agency,
            month,
            year,
            award_amount
        )]
        schema = [
            "awarding_agency", "awarding_sub_agency", "contract_award_type",
            "funding_agency", "funding_sub_agency", "month", "year", "award_amount"
        ]
        df_input = spark.createDataFrame(data, schema=schema)

        # Pass to the pipeline for inference
        pred_df = pipeline_model_reg.transform(df_input)
        if pred_df.rdd.isEmpty():
            flash("No valid data to predict on.", "danger")
        else:
            prediction_value = pred_df.collect()[0]["prediction"]
            flash(f"Regression Prediction (award_amount) = {prediction_value:,.2f}", "success")

    return render_template("regression.html",
                           awarding_agency=awarding_agency,
                           awarding_sub_agency=awarding_sub_agency,
                           contract_award_type=contract_award_type,
                           funding_agency=funding_agency,
                           funding_sub_agency=funding_sub_agency,
                           month=month,
                           year=year)


@app.route("/classification", methods=["GET", "POST"])
def classification():
    """
    Handle classification model inference for award amount classification.

    On GET request:
        Renders the classification form with default values.

    On POST request:
        Processes form data, performs classification using the loaded pipeline model,
        and flashes the classification result with confidence.

    Returns:
        A rendered HTML template for classification or a redirect with flashed messages.
    """
    awarding_agency = "Department of Defense"
    awarding_sub_agency = None
    contract_award_type = None
    funding_agency = None
    funding_sub_agency = None
    month = 1
    year = 2023
    award_amount = 0.0

    if request.method == "POST":
        awarding_agency = request.form.get("awarding_agency", "")
        awarding_sub_agency = request.form.get("awarding_sub_agency", "")
        contract_award_type = request.form.get("contract_award_type", "")
        funding_agency = request.form.get("funding_agency", "")
        funding_sub_agency = request.form.get("funding_sub_agency", "")
        month = int(request.form.get("month", 1))
        year = int(request.form.get("year", 2023))
        award_amount = float(request.form.get("award_amount", 0.0))

        data = [(
            awarding_agency,
            awarding_sub_agency,
            contract_award_type,
            funding_agency,
            funding_sub_agency,
            month,
            year,
            award_amount
        )]
        schema = [
            "awarding_agency", "awarding_sub_agency", "contract_award_type",
            "funding_agency", "funding_sub_agency", "month", "year", "award_amount"
        ]
        df_input = spark.createDataFrame(data, schema=schema)

        pred_df = pipeline_model_class.transform(df_input)
        if pred_df.rdd.isEmpty():
            flash("No valid data to classify.", "danger")
        else:
            row = pred_df.collect()[0]
            predicted_label = int(row["prediction"])
            probs = row["probability"]
            label_str = "HIGH" if predicted_label == 1 else "LOW"
            confidence = probs[predicted_label] * 100
            flash(f"Classification: {label_str} (Confidence = {confidence:.2f}%)", "success")

    return render_template("classification.html",
                           awarding_agency=awarding_agency,
                           awarding_sub_agency=awarding_sub_agency,
                           contract_award_type=contract_award_type,
                           funding_agency=funding_agency,
                           funding_sub_agency=funding_sub_agency,
                           month=month,
                           year=year)


@app.route("/clustering", methods=["GET", "POST"])
def clustering():
    """
    Handle clustering model inference for award data clustering.

    On GET request:
        Renders the clustering form with default values.

    On POST request:
        Processes form data, performs clustering using the loaded pipeline model,
        and flashes the cluster ID result.

    Returns:
        A rendered HTML template for clustering or a redirect with flashed messages.
    """
    awarding_agency = "Department of Defense"
    awarding_sub_agency = None
    contract_award_type = None
    funding_agency = None
    funding_sub_agency = None
    month = 1
    year = 2023
    award_amount = 0.0

    if request.method == "POST":
        awarding_agency = request.form.get("awarding_agency", "")
        awarding_sub_agency = request.form.get("awarding_sub_agency", "")
        contract_award_type = request.form.get("contract_award_type", "")
        funding_agency = request.form.get("funding_agency", "")
        funding_sub_agency = request.form.get("funding_sub_agency", "")
        month = int(request.form.get("month", 1))
        year = int(request.form.get("year", 2023))
        award_amount = float(request.form.get("award_amount", 0.0))

        data = [(
            awarding_agency,
            awarding_sub_agency,
            contract_award_type,
            funding_agency,
            funding_sub_agency,
            month,
            year,
            award_amount
        )]
        schema = [
            "awarding_agency", "awarding_sub_agency", "contract_award_type",
            "funding_agency", "funding_sub_agency", "month", "year", "award_amount"
        ]
        df_input = spark.createDataFrame(data, schema=schema)

        pred_df = pipeline_model_cluster.transform(df_input)
        if pred_df.rdd.isEmpty():
            flash("No valid data for clustering.", "danger")
        else:
            cluster_id = int(pred_df.collect()[0]["prediction"])
            flash(f"Cluster: {cluster_id}", "success")

    return render_template("clustering.html",
                           awarding_agency=awarding_agency,
                           awarding_sub_agency=awarding_sub_agency,
                           contract_award_type=contract_award_type,
                           funding_agency=funding_agency,
                           funding_sub_agency=funding_sub_agency,
                           month=month,
                           year=year,
                           award_amount=award_amount)


@app.route("/download", methods=["GET"])
def download_page():
    """
    Render the download page for fetching data.

    Returns:
        A rendered HTML template for the download page.
    """
    return render_template("download.html")


@app.route("/fetch_data", methods=["POST"])
def fetch_data():
    """
    Fetch data from the USASpending API and insert it into Cassandra.

    This route handles the data fetching process based on user-specified frequency
    and date range. It fetches data in batches, handles pagination, and inserts
    records into the Cassandra database. It also records the download history.

    Returns:
        A redirect to the download page with flashed messages indicating the status.
    """
    frequency = request.form.get('frequency')
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        if start_date > end_date:
            flash("Start date must be before end date.", "danger")
            return redirect(url_for("download_page"))
    except ValueError:
        flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
        return redirect(url_for("download_page"))

    download_start = datetime.now()
    insert_count = 0
    batch_size = 2000  # Records per batch
    page_size = 100  # Records per page as per response
    pages_per_batch = batch_size // page_size  # Number of pages per batch
    logger.info("Data fetching frequency: %s, Pages per batch: %d", frequency, pages_per_batch)

    has_next = True
    page = 1
    batch_counter = 0

    while has_next:
        try:
            payload = {
                "filters": {
                    "time_period": [{
                        "start_date": start_date.strftime('%Y-%m-%d'),
                        "end_date": end_date.strftime('%Y-%m-%d')
                    }],
                    "award_type_codes": ["A", "B", "C"]
                },
                "fields": [
                    "Award ID",
                    "Recipient Name",
                    "Start Date",
                    "End Date",
                    "Award Amount",
                    "Awarding Agency",
                    "Awarding Sub Agency",
                    "Contract Award Type",
                    "Funding Agency",
                    "Funding Sub Agency",
                    "Description",
                    "Last Modified Date",
                    "Base Obligation Date"
                ],
                "limit": page_size,
                "page": page
            }

            api_url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
            # Make the API request
            response = requests.post(api_url, json=payload, timeout=60)  # Increased timeout
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            page_metadata = data.get("page_metadata", {})
            has_next = page_metadata.get("hasNext", False)

            if not results:
                break

            # Insert fetched records into Cassandra
            for award in results:
                award_id = award.get('Award ID')
                recipient_name = award.get('Recipient Name')
                start_dt = award.get('Start Date')
                end_dt = award.get('End Date')
                award_amount = award.get('Award Amount')
                awarding_agency = award.get('Awarding Agency')
                awarding_sub_agency = award.get('Awarding Sub Agency')
                contract_award_type = award.get('Contract Award Type')
                funding_agency = award.get('Funding Agency')
                funding_sub_agency = award.get('Funding Sub Agency')
                description = award.get('Description')
                last_modified_date = award.get('Last Modified Date')
                base_obligation_date = award.get('Base Obligation Date')

                # Convert dates
                start_date_converted = to_date(start_dt)
                end_date_converted = to_date(end_dt)
                lmd = to_date(last_modified_date)
                base_obligation_date_converted = to_date(base_obligation_date)

                # Convert award_amount to float
                aa = None
                if award_amount is not None:
                    try:
                        aa = float(award_amount)
                    except ValueError:
                        aa = None

                # Insert into Cassandra
                cassandra_session.execute(
                    """
                    INSERT INTO awards (
                        award_id, recipient_name, start_date, end_date, award_amount,
                        awarding_agency, awarding_sub_agency, contract_award_type,
                        funding_agency, funding_sub_agency, description,
                        last_modified_date, base_obligation_date
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        award_id, recipient_name, start_date_converted, end_date_converted, aa,
                        awarding_agency, awarding_sub_agency, contract_award_type,
                        funding_agency, funding_sub_agency, description,
                        lmd, base_obligation_date_converted
                    )
                )
                insert_count += 1
                batch_counter += 1

                if batch_counter >= batch_size:
                    batch_counter = 0
                    logger.info(f"Inserted {insert_count} records.")

                    # Wait for 3 to 5 seconds
                    wait_time = random.randint(3, 5)
                    logger.info(f"Waiting for {wait_time} seconds before next batch...")
                    time.sleep(wait_time)
            page += 1

        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Connection error encountered: {ce}")
            logger.error("Retrying after a 3 seconds delay...")
            time.sleep(3)
            continue
        except requests.exceptions.Timeout as te:
            logger.error(f"Request timed out: {te}")
            logger.error("Retrying after a 3 seconds delay...")
            time.sleep(3)
            continue
        except requests.exceptions.HTTPError as he:
            logger.error(f"HTTP error occurred: {he}")
            flash(f"HTTP error: {he}", "danger")
            return redirect(url_for("download_page"))
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            flash(f"An unexpected error occurred: {e}", "danger")
            return redirect(url_for("download_page"))

    # Record download metadata
    download_end = datetime.now()
    download_id = uuid.uuid4()
    try:
        cassandra_session.execute(
            """
            INSERT INTO download_history (download_id, start_time, end_time, number_of_rows)
            VALUES (%s, %s, %s, %s)
            """,
            (download_id, download_start, download_end, insert_count)
        )
    except Exception as e:
        logger.error(f"Error recording download history: {e}")
        flash("Data fetched but failed to record download history.", "warning")
        return redirect(url_for("download_page"))

    flash(f"Fetched and inserted records into Cassandra. Download log recorded.", "success")
    return redirect(url_for("download_page"))


def to_date(d):
    """
    Convert a string to a date object.

    Args:
        d (str): Date string in the format 'YYYY-MM-DD'.

    Returns:
        datetime.date or None: Converted date object if successful, else None.
    """
    if d:
        try:
            return datetime.strptime(d, '%Y-%m-%d').date()
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    """
    Runs the Flask app in debug mode.
    """
    app.run(debug=True)
