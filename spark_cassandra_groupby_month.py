from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, month as month_func


def main():
    spark = (
        SparkSession.builder
        .appName("GroupByASAAndMonth")
        .config("spark.cassandra.connection.host", "127.0.0.1")
        .config("spark.cassandra.connection.port", "9042")
        .getOrCreate()
    )

    # Read from the "awards" table in Cassandra
    df_awards = (
        spark.read
        .format("org.apache.spark.sql.cassandra")
        .options(table="awards", keyspace="usaspending_data")
        .load()
    )

    # Extract the month from the desired date column
    df_awards_with_month = df_awards.withColumn("month", month_func("start_date"))

    # Group by awarding_sub_agency and month, then sum the award_amount
    df_grouped = (
        df_awards_with_month
        .groupBy("awarding_sub_agency", "month")
        .agg(_sum("award_amount").alias("total_award_amount"))
    )

    # Write results to the Cassandra table: awarding_sub_agency_by_month
    (
        df_grouped
        .write
        .format("org.apache.spark.sql.cassandra")
        .options(table="awarding_sub_agency_by_month", keyspace="usaspending_data")
        .mode("append")  # or "overwrite" if you want to truncate first
        .save()
    )

    spark.stop()


if __name__ == "__main__":
    main()
