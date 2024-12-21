"""
BigData & Tools USASpending Application
===========================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

This module is a core component of the **BigData & Tools USASpending Application**. It is responsible for:

- **Data Ingestion:** Connecting to a Cassandra database to retrieve USASpending award data.
- **Data Preprocessing:** Cleaning the data by handling missing values, extracting temporal features (month and year), and filtering out invalid records.
- **Correlation Analysis:** Performing correlation computations on numerical and categorical features and generating a heatmap visualization to identify relationships between variables.
- **Machine Learning Pipelines:** Building and training pipelines for three machine learning tasks:
- **Regression:** Predicting award amounts using Linear Regression.
- **Classification:** Categorizing awards into high or low based on the median award amount using Logistic Regression.
- **Clustering:** Grouping awards into clusters using K-Means clustering.
- **Model Evaluation and Saving:** Evaluating the performance of the trained models, logging the results, and saving both the evaluation metrics and the trained pipeline models for future use.

The module leverages PySpark for distributed data processing and machine learning, ensuring scalability and efficiency when handling large datasets. Additionally, it utilizes Matplotlib for generating visualizations and incorporates custom configuration and logging setups to maintain robust and maintainable code.

** Author: ** Buddhi Ayesha
** Date: ** 2024 - 12 - 21
"""

import os

import matplotlib

from app import config
from app.logging_config import setup_logging

logger = setup_logging()

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    month as month_func,
    year as year_func,
    when,
    col,
    lit
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.stat import Correlation


def main():
    """
    Executes the USASpending data processing and machine learning pipeline.

    This function performs the following steps:

    1. **Setup Environment:**
       - Creates necessary output directories if they do not exist.
       - Initializes a Spark session with Cassandra configurations.

    2. **Data Ingestion:**
       - Reads award data from a Cassandra table into a Spark DataFrame.

    3. **Data Preprocessing:**
       - Drops rows with missing values in required columns.
       - Extracts month and year from the `start_date`.
       - Filters out records with non-positive award amounts.
       - Identifies valid categorical columns based on distinct value counts.

    4. **Correlation Analysis:**
       - Indexes categorical columns.
       - Assembles numeric features for correlation computation.
       - Generates and saves a correlation heatmap if sufficient data is available.

    5. **Pipeline Construction:**
       - Builds separate pipelines for regression, classification, and clustering tasks.
       - Includes stages for indexing, encoding, feature assembling, and model training.

    6. **Model Training and Evaluation:**
       - Splits data into training and testing sets.
       - Fits each pipeline on the training data.
       - Evaluates regression and classification models using RMSE and AUC metrics, respectively.
       - Performs clustering using K-Means and retrieves cluster centers.

    7. **Results Saving:**
       - Compiles summary statistics and model evaluation metrics.
       - Writes the compiled information to a text file.
       - Saves the trained pipeline models for future use.

    **Logging:**
    - Logs warnings for columns with insufficient distinct values.
    - Logs warnings if correlation analysis or model evaluations are skipped due to lack of data.

    **Output:**
    - Correlation heatmap saved as `static/images/correlation_heatmap.png`.
    - Model training information saved in `outputs/model_training_info.txt`.
    - Trained pipeline models saved in the `outputs` directory.

    **Dependencies:**
    - Requires a running Cassandra instance with the appropriate keyspace and table.
    - Utilizes PySpark for data processing and machine learning.
    - Uses Matplotlib for visualization.
    - Custom configuration and logging modules (`config`, `logging_config`).
    """

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    spark = (
        SparkSession.builder
        .appName("MultiUseCase_ML_Models_WithPipelines")
        .config("spark.cassandra.connection.host", config.CASSANDRA_HOST)
        .config("spark.cassandra.connection.port", config.CASSANDRA_PORT)
        .getOrCreate()
    )

    # Read data from Cassandra
    df_awards = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awards", keyspace="usaspending_data")
        .load()
    )

    # Data Preprocessing
    required_cols = [
        "award_amount",
        "start_date",
        "awarding_agency",
        "awarding_sub_agency",
        "contract_award_type",
        "funding_agency",
        "funding_sub_agency"
    ]
    df_awards = df_awards.dropna(subset=required_cols)

    df_awards = df_awards.withColumn("month", month_func("start_date"))
    df_awards = df_awards.withColumn("year", year_func("start_date"))
    df_awards = df_awards.filter(col("award_amount") > 0)

    cat_columns = [
        "awarding_agency",
        "awarding_sub_agency",
        "contract_award_type",
        "funding_agency",
        "funding_sub_agency"
    ]

    valid_cat_columns = []
    for c in cat_columns:
        distinct_count = df_awards.select(c).distinct().count()
        if distinct_count < 2:
            logger.warn(f"Column '{c}' has {distinct_count} distinct value(s). Skipping it.")
        else:
            valid_cat_columns.append(c)

    cat_columns = valid_cat_columns

    # Correlation Analysis
    numeric_cols = ["award_amount", "month", "year"]
    for c in cat_columns:
        numeric_cols.append(f"{c}_index")

    df_indexed = df_awards
    for cat_col in cat_columns:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_index",
            handleInvalid="skip"
        )
        df_indexed = indexer.fit(df_indexed).transform(df_indexed)

    # Compute correlation
    from pyspark.ml.feature import VectorAssembler
    assembler_for_corr = VectorAssembler(inputCols=numeric_cols, outputCol="corr_features")
    df_corr = assembler_for_corr.transform(df_indexed).select("corr_features")

    if df_corr.rdd.isEmpty():
        logger.warn("No data rows remain for correlation. Skipping correlation heatmap.")
    else:
        corr_matrix = Correlation.corr(df_corr, "corr_features", method="pearson").head()[0]
        corr_array = corr_matrix.toArray()

        plt.figure(figsize=(8, 6))
        plt.imshow(corr_array, cmap="viridis", interpolation="nearest")

        # Annotate the matrix with correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text_val = f"{corr_array[i, j]:.2f}"
                plt.text(
                    j, i, text_val,
                    ha="center", va="center",
                    color="white" if abs(corr_array[i, j]) > 0.5 else "black",
                    fontsize=8
                )

        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title("Correlation Heatmap with Annotations")
        plt.tight_layout()
        plt.savefig("static/images/correlation_heatmap.png")
        plt.close()

    # Build Pipelines
    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_index",
            handleInvalid="skip"
        ) for c in cat_columns
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_index" for c in cat_columns],
        outputCols=[f"{c}_vec" for c in cat_columns]
    )

    reg_feature_cols = [f"{c}_vec" for c in cat_columns] + ["month", "year"]
    assembler_reg = VectorAssembler(
        inputCols=reg_feature_cols,
        outputCol="features_reg"
    )
    lr = LinearRegression(featuresCol="features_reg", labelCol="award_amount")
    pipeline_reg = Pipeline(stages=indexers + [encoder, assembler_reg, lr])

    stats = df_awards.approxQuantile("award_amount", [0.5], 0.001)
    median_award_amount = stats[0] if stats else 100000
    df_awards = df_awards.withColumn(
        "label_class",
        when(col("award_amount") > lit(median_award_amount), 1.0).otherwise(0.0)
    )

    class_feature_cols = reg_feature_cols
    assembler_class = VectorAssembler(
        inputCols=class_feature_cols,
        outputCol="features_class"
    )
    logreg = LogisticRegression(featuresCol="features_class", labelCol="label_class", maxIter=20)
    pipeline_class = Pipeline(stages=indexers + [encoder, assembler_class, logreg])
    cluster_feature_cols = reg_feature_cols + ["award_amount"]
    assembler_cluster = VectorAssembler(
        inputCols=cluster_feature_cols,
        outputCol="features_cluster"
    )
    kmeans = KMeans(featuresCol="features_cluster", k=5, seed=42)

    pipeline_clustering = Pipeline(stages=indexers + [encoder, assembler_cluster, kmeans])

    # Fit Each Pipeline
    # Regression
    train_reg, test_reg = df_awards.randomSplit([0.8, 0.2], seed=42)
    pipeline_model_reg = pipeline_reg.fit(train_reg)

    # Evaluate
    pred_reg = pipeline_model_reg.transform(test_reg)
    if pred_reg.rdd.isEmpty():
        logger.warn("No test data for Regression after pipeline. Skipping evaluation.")
        lr_rmse = float("nan")
    else:
        lr_evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol="award_amount",
            metricName="rmse"
        )
        lr_rmse = lr_evaluator.evaluate(pred_reg)

    # Classification
    train_class, test_class = df_awards.randomSplit([0.8, 0.2], seed=42)
    pipeline_model_class = pipeline_class.fit(train_class)

    pred_class = pipeline_model_class.transform(test_class)
    if pred_class.rdd.isEmpty():
        logger.warn("No test data for Classification. Skipping evaluation.")
        logreg_auc = float("nan")
    else:
        bc_evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label_class",
            metricName="areaUnderROC"
        )
        logreg_auc = bc_evaluator.evaluate(pred_class)

    # Clustering
    pipeline_model_cluster = pipeline_clustering.fit(df_awards)
    kmeans_model = pipeline_model_cluster.stages[-1]
    kmeans_centers = kmeans_model.clusterCenters()

    # Save Info & Models
    summary_stats_reg = train_reg.select("award_amount", "month", "year").describe().collect()
    stats_str = "=== Training Data Summary (Regression) ===\n"
    for row in summary_stats_reg:
        stats_str += str(row.asDict()) + "\n"

    # Add regression result
    training_info_str = "\n=== MODEL #1: REGRESSION (Predict award_amount) ===\n"
    training_info_str += f"Linear Regression RMSE: {lr_rmse:.4f}\n"

    # Add classification result
    training_info_str += "\n=== MODEL #2: CLASSIFICATION (High vs Low) ===\n"
    training_info_str += f"Median award_amount threshold: {median_award_amount}\n"
    training_info_str += f"Logistic Regression AUC: {logreg_auc:.4f}\n"

    # Add clustering result
    training_info_str += "\n=== MODEL #3: CLUSTERING (K-Means) ===\n"
    training_info_str += "Number of clusters (k=5)\nCluster Centers:\n"
    for i, center in enumerate(kmeans_centers):
        training_info_str += f"  Cluster {i}: {center}\n"

    # Write model logs to file
    with open("outputs/model_training_info.txt", "w") as f:
        f.write(stats_str)
        f.write(training_info_str)

    # Save pipeline models
    pipeline_model_reg.save("outputs/pipeline_regression")
    pipeline_model_class.save("outputs/pipeline_classification")
    pipeline_model_cluster.save("outputs/pipeline_clustering")

    spark.stop()


if __name__ == "__main__":
    main()
