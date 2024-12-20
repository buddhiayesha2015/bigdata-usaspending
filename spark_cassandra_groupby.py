from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum


def main():
    spark = (
        SparkSession.builder
        .appName("GroupByASAAndRecipient")
        .config("spark.cassandra.connection.host", "127.0.0.1")
        .config("spark.cassandra.connection.port", "9042")
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
    main()
