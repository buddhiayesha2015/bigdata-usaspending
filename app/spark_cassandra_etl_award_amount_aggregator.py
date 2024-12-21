"""
BigData & Tools USASpending Application
============================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

This module serves as the entry point for the **BigData & Tools USASpending Application**.
It performs an Extract, Transform, Load (ETL) process using Apache Spark to aggregate award amounts based on specified
criteria—either by recipient name or awarding sub-agency. The application connects to a Cassandra database to read and
write data, ensuring efficient handling of large datasets. Command-line arguments allow users to specify the aggregation
type, and comprehensive logging facilitates monitoring and debugging. This module is designed for scalability and
maintainability, adhering to best practices in big data processing.

**Author:** Buddhi Ayesha
**Date:** 2024-12-21
"""

import argparse
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum

from app import config
from app.logging_config import setup_logging

logger = setup_logging()


def spark_cassandra_aggregator(aggregator):
    """
    Executes the ETL process to aggregate award amounts based on the specified aggregator.

    This function initializes a Spark session, reads data from Cassandra tables,
    performs aggregation based on the provided aggregator, and writes the results
    back to a Cassandra table.

    Args:
        aggregator (str): The column to aggregate by. Supported values are
                          'recipient_name' and 'awarding_sub_agency'.

    Raises:
        SystemExit: If an unsupported aggregator is provided.
    """
    if aggregator == "recipient_name":
        geo_table = "recipient_name_with_geo"
        output_table = "total_award_amount_by_recipient"
    elif aggregator == "awarding_sub_agency":
        geo_table = "awarding_sub_agency_with_geo"
        output_table = "total_award_amount_by_awarding_sub_agency"
    else:
        logger.error(f"Unsupported aggregator: {aggregator}")
        sys.exit(1)

    spark = (
        SparkSession.builder
        .appName("CassandraSparkETL")
        .config("spark.cassandra.connection.host", config.CASSANDRA_HOST)
        .config("spark.cassandra.connection.port", config.CASSANDRA_PORT)
        .getOrCreate()
    )

    # Read from Cassandra: geo table and awards table
    df_geo = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table=geo_table, keyspace="usaspending_data")
        .load()
    )

    df_awards = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awards", keyspace="usaspending_data")
        .load()
    )

    # Perform the join on aggregator column
    df_joined = (
        df_geo.join(df_awards, aggregator, "inner")
        .groupBy(aggregator, "latitude", "longitude")
        .agg(_sum("award_amount").alias("total_award_amount"))
    )

    # Write to the new Cassandra table
    (
        df_joined
        .write
        .format("org.apache.spark.sql.cassandra")
        .options(table=output_table, keyspace="usaspending_data")
        .mode("append")
        .save()
    )

    spark.stop()


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Spark ETL with aggregator argument"
    )
    parser.add_argument(
        "--aggregator",
        required=True,
        choices=["recipient_name", "awarding_sub_agency"],
        help="Aggregator column: 'recipient_name' or 'awarding_sub_agency'"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    spark_cassandra_aggregator(args.aggregator)
