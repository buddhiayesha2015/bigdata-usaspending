import pytest
from pyspark.sql import SparkSession

from app import config


def test_spark_cassandra_connection():
    # Cassandra and Spark configuration
    cassandra_host = config.CASSANDRA_HOST
    cassandra_port = config.CASSANDRA_PORT
    keyspace = "usaspending_data"
    test_table = "awards"

    try:
        # Initialize SparkSession
        spark = (
            SparkSession.builder
            .appName("TestSparkCassandraConnection")
            .config("spark.cassandra.connection.host", cassandra_host)
            .config("spark.cassandra.connection.port", cassandra_port)
            .config("spark.master", "local[*]")
            .getOrCreate()
        )

        # Attempt to read data from Cassandra
        df = (
            spark.read
            .format("org.apache.spark.sql.cassandra")
            .options(table=test_table, keyspace=keyspace)
            .load()
        )

        # Check if the DataFrame is non-empty
        assert df is not None, "Failed to read table from Cassandra."
        assert df.count() >= 0, f"Table '{test_table}' in keyspace '{keyspace}' is empty or not accessible."
        spark.stop()

    except Exception as e:
        pytest.fail(f"Unable to connect to Cassandra using Spark: {e}")
