"""
BigData & Tools USASpending Application
===========================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

The **BigData & Tools USASpending Application** is a comprehensive system designed to process, analyze, and visualize USASpending data. Leveraging powerful big data technologies such as Apache Spark and Cassandra, the application offers robust functionalities including:

- **Data Processing & Aggregation:** Extracts, transforms, and loads award data, aggregating amounts by recipient and sub-agency.
- **Geolocation Data Retrieval:** Processes recipient names to obtain geographical coordinates using the Nominatim API.
- **Machine Learning:** Implements regression, classification, and clustering models to analyze award trends and predict outcomes.
- **API & Visualization:** Provides endpoints for data visualization dashboards and machine learning inference, enabling interactive data exploration.
- **ETL Pipelines:** Automates the Extract, Transform, Load processes to ensure efficient and scalable data handling.
- **Logging & Configuration:** Incorporates comprehensive logging and configurable settings to maintain robust and maintainable operations.

This application is built with scalability and efficiency in mind, ensuring seamless handling of large datasets while providing insightful analytics and visualizations for informed decision-making.

**Author:** Buddhi Ayesha
**Date:** 2024-12-21
"""

__pdoc__ = {
    "app.config": False,
    "app.fetch_awarding_sub_agency_geo_data": False,
    "app.logging_config": False,
    "app.spark_cassandra_groupby_month": False
}
