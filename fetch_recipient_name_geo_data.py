from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from collections import Counter
import requests
import time

import config

try:
    from tqdm import tqdm

    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def get_geolocation(name):
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


def main():
    cluster = Cluster(contact_points=[config.CASSANDRA_HOST], port=config.CASSANDRA_PORT)
    session = cluster.connect("usaspending_data")

    print("Connected to Cassandra cluster.")

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

    print("Counting completed.")
    print(f"Total unique recipient_names: {len(recipient_counter)}\n")

    top_n = 5
    print(f"Top {top_n} recipients by award count:")
    for name, count in recipient_counter.most_common(top_n):
        print(f"{name}: {count}")

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
            print(f"Collected {successful_geo_count}/{max_geo}: {name}")
        time.sleep(1)

    # Insert geolocation data into Cassandra
    if geo_data_list:
        for record in geo_data_list:
            session.execute(insert_statement, record)
        print(
            f"\nGeolocation data for {successful_geo_count} recipients inserted into 'recipient_name_with_geo' table.")
    cluster.shutdown()


if __name__ == "__main__":
    main()
