"""
BigData & Tools USASpending Application
===========================================

![USASpending Explorer](../assets/usaspending_explorer.svg)

### Module Overview

Processes recipient names from the Cassandra `awards` table, counts their occurrences, retrieves geolocation data for
top recipients using the Nominatim API, and stores the geolocation information back into the Cassandra database.

**Author:** Buddhi Ayesha
**Date:** 2024-12-21
"""

import time
from collections import Counter

import requests
from cassandra.cluster import Cluster

from app import config
from app.logging_config import setup_logging

logger = setup_logging()

try:
    from tqdm import tqdm

    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def get_geolocation(name):
    """
    Retrieve the geographical coordinates (latitude and longitude) for a given location name.

    This function queries the Nominatim API from OpenStreetMap to obtain the geolocation data.

    Args:
        name (str): The name of the location to geocode.

    Returns:
        Tuple[Optional[float], Optional[float]]:
            A tuple containing the latitude and longitude as floats if found,
            otherwise (None, None).

    Example:<br>
        `>>> lat, lon = get_geolocation("New York")`<br>
        `>>> print(lat, lon)`<br>
        `40.7127281 -74.0060152`
    """
    url = 'https://nominatim.openstreetmap.org/search'
    params = {'q': name, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'CassandraBackupScript/1.0'}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data.get('lat')), float(data.get('lon'))
    except requests.RequestException:
        pass
    return None, None


def fetch_recipient():
    """
    This function for processing recipient names and storing their geolocations.

    This function connects to a Cassandra cluster, retrieves recipient names from the
    'awards' table, counts the occurrences of each recipient, fetches geolocation data
    for the top recipients, and inserts the geolocation data into the 'recipient_name_with_geo'
    table.

    Steps:
        1. Connect to the Cassandra cluster using configuration parameters.
        2. Fetch all `recipient_name` entries from the `awards` table.
        3. Count the occurrences of each unique `recipient_name`.
        4. Log the total number of unique recipient names and the top recipients by award count.
        5. Fetch geolocation data for the top recipients (up to a maximum limit).
        6. Insert the collected geolocation data into the Cassandra table.
        7. Shutdown the Cassandra cluster connection.

    Raises:
        None
    """
    cluster = Cluster(contact_points=[config.CASSANDRA_HOST], port=config.CASSANDRA_PORT)
    session = cluster.connect("usaspending_data")

    logger.info("Connected to Cassandra cluster.")

    # Fetch all recipient_names
    query = "SELECT recipient_name FROM awards"
    statement = session.prepare(query)
    results = session.execute(statement)

    # Count unique recipient_names
    recipient_counter = Counter()
    iterator = tqdm(results, desc="Processing recipient_names") if USE_TQDM else results

    for row in iterator:
        recipient_name = row.recipient_name
        if recipient_name:
            recipient_counter[recipient_name] += 1

    logger.info("Counting completed.")
    logger.info(f"Total unique recipient_names: {len(recipient_counter)}\n")

    top_n = 5
    logger.info(f"Top {top_n} recipients by award count:")
    for name, count in recipient_counter.most_common(top_n):
        logger.info(f"{name}: {count}")

    insert_query = """
    INSERT INTO recipient_name_with_geo (recipient_name, latitude, longitude)
    VALUES (?, ?, ?);
    """
    insert_statement = session.prepare(insert_query)

    geo_data_list = []
    successful_geo_count = 0
    max_geo = 10

    for name, count in recipient_counter.most_common():
        if successful_geo_count >= max_geo:
            break
        lat, lon = get_geolocation(name)
        if lat is not None and lon is not None:
            geo_data_list.append((name, lat, lon))
            successful_geo_count += 1
            logger.info(f"Collected {successful_geo_count}/{max_geo}: {name}")
        time.sleep(1)

    # Insert geolocation data into Cassandra
    if geo_data_list:
        for record in geo_data_list:
            session.execute(insert_statement, record)
        logger.info(
            f"\nGeolocation data for {successful_geo_count} recipients inserted into 'recipient_name_with_geo' table.")
    cluster.shutdown()


if __name__ == "__main__":
    fetch_recipient()
