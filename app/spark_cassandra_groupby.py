"""
BigData & Tools USASpending Application
============================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

This module performs data aggregation on the USASpending data stored in Cassandra.
It reads the "awards" table, groups the data by `awarding_sub_agency` and `recipient_name`,
sums the `award_amount`, and writes the aggregated results to a new Cassandra table
`awarding_sub_agency_by_recipient`.

Dependencies:
- pyspark

Usage:
    Run the module as a standalone script to execute the data processing pipeline.

**Author:** Buddhi Ayesha
**Date:** 2024-12-21
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum

from app import config


def awarding_agency_by_recipient():
    """
    This function to aggregate award amounts by awarding sub-agency and recipient.

    This function initializes a Spark session with Cassandra configurations,
    reads data from the "awards" table in the "usaspending_data" keyspace,
    performs a group by operation on `awarding_sub_agency` and `recipient_name`,
    sums the `award_amount`, and writes the aggregated data to the
    "awarding_sub_agency_by_recipient" table in Cassandra.

    Steps:
        1. Initialize SparkSession with Cassandra connection settings.
        2. Load data from the "awards" table.
        3. Group data by `awarding_sub_agency` and `recipient_name`.
        4. Aggregate the `award_amount` using sum.
        5. Write the aggregated data to the new Cassandra table.
        6. Stop the Spark session.

    Raises:
        Any exceptions raised during Spark operations will propagate upwards.
    """
    spark = (
        SparkSession.builder
        .appName("GroupByASAAndRecipient")
        .config("spark.cassandra.connection.host", config.CASSANDRA_HOST)
        .config("spark.cassandra.connection.port", config.CASSANDRA_PORT)
        .getOrCreate()
    )

    # Read from the "awards" table
    df_awards = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awards", keyspace="usaspending_data")
        .load()
    )

    # Group by awarding_sub_agency AND recipient_name, then sum the award_amount
    df_grouped = (
        df_awards
        .groupBy("awarding_sub_agency", "recipient_name")
        .agg(_sum("award_amount").alias("total_award_amount"))
    )

    # Write results to the new Cassandra table: awarding_sub_agency_by_recipient
    (
        df_grouped
        .write
        .format("org.apache.spark.sql.cassandra")
        .options(table="awarding_sub_agency_by_recipient", keyspace="usaspending_data")
        .mode("append")
        .save()
    )

    spark.stop()


if __name__ == "__main__":
    awarding_agency_by_recipient()
