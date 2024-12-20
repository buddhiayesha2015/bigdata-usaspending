import argparse
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum

import config


def main(aggregator):
    if aggregator == "recipient_name":
        geo_table = "recipient_name_with_geo"
        output_table = "total_award_amount_by_recipient"
    elif aggregator == "awarding_sub_agency":
        geo_table = "awarding_sub_agency_with_geo"
        output_table = "total_award_amount_by_awarding_sub_agency"
    else:
        print(f"ERROR: Unsupported aggregator: {aggregator}")
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


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Spark ETL with aggregator argument")
    parser.add_argument("--aggregator", required=True,
                        help="Aggregator column: 'recipient_name' or 'awarding_sub_agency'")
    args = parser.parse_args()

    main(args.aggregator)
