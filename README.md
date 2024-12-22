# Big Data & Tools - USASpending

## Overview
The **Big Data & Tools - USASpending** project is a comprehensive application designed to process, analyze, and visualize financial data from the USASpending API. It leverages distributed big data technologies such as **Apache Spark** and **Cassandra**, alongside machine learning models, to provide actionable insights into government spending.

---

## Features
- **Data Processing:** 
  - Extract, transform, and load (ETL) data using Apache Spark and Cassandra.
  - Group and aggregate data by recipient, sub-agency, and time intervals.

- **Machine Learning Integration:**
  - Perform clustering, classification, regression, and correlation analysis on processed data.

- **Web Application:**
  - Visualize data and insights using Flask.
  - Provide interactive dashboards.

---

## Dependencies

### Software & Libraries
- **Apache Spark:** 3.4.4  
- **Apache Cassandra:** 4.1.7  
- **Scala:** 2.12.17  
- **Python:** 3.12.8  
- **pytest:** 8.3.4  
- **OpenJDK:** 11.0.25  
- **Flask:** 3.1.0  

### Operating System
- **Ubuntu 22.04.5 LTS**  

---

## Folder Structure

```
├── app
│   ├── config.py                                        # Configuration settings for the application
│   ├── fetch_awarding_sub_agency_geo_data.py            # Fetches geolocation data for awarding sub-agencies
│   ├── fetch_recipient_name_geo_data.py                 # Fetches geolocation data for recipients
│   ├── logging_config.py                                # Logging setup for the project
│   ├── machine_learning_models.py                       # Train Machine Learning models for clustering, classification, and regression
│   ├── ml_app.py                                        # Flask application for machine learning and visualization
│   ├── spark_cassandra_etl_award_amount_aggregator.py   # Processes award data and aggregates by recipient
│   ├── spark_cassandra_groupby.py                       # Groups data by various criteria
│   ├── spark_cassandra_groupby_month.py                 # Groups data by months
│   └── __init__.py
├── assets
│   └── usaspending_explorer.svg  # Visualization assets
├── cql
│   └── setup_usaspending.cql     # Cassandra schema setup script
├── docs
│   └── (HTML Documentation files)
├── logs
│   └── (Log files generated during application execution)
├── outputs
│   ├── pipeline_clustering       # Clustering pipeline outputs
│   ├── pipeline_classification   # Classification pipeline outputs
│   ├── pipeline_regression       # Regression pipeline outputs
│   └── model_training_info.txt   # Logs of machine learning model training
├── static
│   └── (Static assets like images and CSS)
├── templates
│   └── (HTML templates for the web app)
├── tests
│   ├── test_ml_app.py
│   └── test_spark_cassandra.py
└── README.md
```

---

## Prerequisites

1. **Python:** Version 3.8 or above.
2. **Cassandra:** A working Cassandra instance.
3. **Apache Spark:** Installed and configured.
4. **Dependencies:** Install Python dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

---

## Setup and Usage

### Step 1: Configure the Application
Edit the `config.py` file to update Cassandra connection settings and other configurations.

### Step 2: Set Up Cassandra Schema
Run the following command to set up the necessary schema in Cassandra:
```bash
cqlsh -f cql/setup_usaspending.cql
```

### Step 3: Run Data Processing
Execute the Spark ETL scripts to process and load data into Cassandra:

Before running the scripts, ensure Spark can locate the `app/` folder by adding the current directory to your Python path. In a terminal, navigate to the project folder (which contains `app/`) and run:

```bash
cd PROJECT_FOLDER
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

```bash
spark-submit \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.0,com.github.jnr:jnr-posix:3.1.20 \
  app/spark_cassandra_etl_award_amount_aggregator.py \
  --aggregator recipient_name
```

### Step 4: Start the Web Application
Launch the Flask web app to interact with the data and machine learning models:
```bash
python -m app.ml_app
```

**Note:** Ensure that the `BDT_SECRET_KEY` environment variable is set. This key is used to configure the application's secret key.
```python
# Get the secret key from the environment variable
app.secret_key = os.getenv("BDT_SECRET_KEY")
```

### Step 5: Access the Application
Visit `http://127.0.0.1:5000/dashboard` in your browser to explore the dashboards and visualizations.

---

## Testing
Run the test suite to verify the functionality of the application:
```bash
pytest tests/
```

---

## Key Scripts

### 1. `spark_cassandra_etl_award_amount_aggregator.py`
Processes award data and aggregates it by recipient.

### 2. `fetch_recipient_name_geo_data.py`
Fetches geolocation data for recipients using external APIs.

### 3. `machine_learning_models.py`
Contains machine learning models for clustering, classification, and regression.

### 4. `ml_app.py`
A Flask application for serving dashboards and machine learning endpoints.

---

## Contributors
- **Author:** Buddhi Ayesha

---

## License

This project is licensed under the Apache License 2.0. You may obtain a copy of the License at:

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Acknowledgements
- **USASpending API** for financial data.
- **Apache Spark** and **Cassandra** for data processing and storage.
- **Flask** for the web application framework.