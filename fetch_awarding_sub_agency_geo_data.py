import time
from collections import Counter

import requests
from cassandra.cluster import Cluster

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
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect("usaspending_data")
    print("Connected to Cassandra cluster.")

    query = "SELECT awarding_sub_agency FROM awards"
    statement = session.prepare(query)
    results = session.execute(statement)

    # Count unique awarding_sub_agency
    awarding_sub_agency_counter = Counter()
    iterator = tqdm(results, desc="Processing awarding_sub_agency") if USE_TQDM else results

    for row in iterator:
        awarding_sub_agency = row.awarding_sub_agency
        if awarding_sub_agency:
            awarding_sub_agency_counter[awarding_sub_agency] += 1

    print("Counting completed.")
    print(f"Total unique awarding_sub_agency: {len(awarding_sub_agency_counter)}\n")
    print("Awarding Sub-Agencies and their counts:")
    for name, count in awarding_sub_agency_counter.most_common():
        print(f"{name}: {count}")

    insert_query = """
    INSERT INTO awarding_sub_agency_with_geo (awarding_sub_agency, latitude, longitude)
    VALUES (?, ?, ?);
    """
    insert_statement = session.prepare(insert_query)

    # Collect geolocation data for awarding_sub_agency
    geo_data_list = []
    successful_geo_count = 0

    for name, count in awarding_sub_agency_counter.most_common():
        lat, lon = get_geolocation(name)
        if lat is not None and lon is not None:
            geo_data_list.append((name, lat, lon))
            successful_geo_count += 1
            print(f"Collected geolocation for: {name}")
        time.sleep(1)

    if geo_data_list:
        for record in geo_data_list:
            session.execute(insert_statement, record)
        print(
            f"\nGeolocation data for {successful_geo_count} awarding_sub_agencies inserted into 'awarding_sub_agency_with_geo' table.")

    cluster.shutdown()


if __name__ == "__main__":
    main()
