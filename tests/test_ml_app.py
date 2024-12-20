from unittest.mock import MagicMock

import pytest
import requests
from cassandra.cluster import Cluster

from app import config
from app.ml_app import app


class MockResponse:
    def __init__(self, json_data=None, status_code=200):
        self.json_data = json_data or {}
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_dashboard_route(client):
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"USASpending" in response.data


def test_fetch_data_invalid_date(client):
    response = client.post(
        '/fetch_data',
        data={'frequency': 'now', 'start_date': '2024-12-21', 'end_date': '2023-12-21'},
        follow_redirects=False
    )
    assert response.status_code == 302
    # Check if the response redirects to /download
    assert "/download" in response.headers['Location']


def test_fetch_data_valid_request(client, mocker):
    # Mock the logger to prevent actual logging during tests
    mock_logger = mocker.patch('app.ml_app.logger')

    # Mock the API response
    mock_api_response = MockResponse(
        json_data={
            "results": [
                {
                    "Award ID": "12345",
                    "Recipient Name": "Test Recipient",
                    "Start Date": "2023-12-01",
                    "End Date": "2023-12-21",
                    "Award Amount": 100000.0,
                    "Awarding Agency": "Test Agency",
                    "Awarding Sub Agency": "Test Sub Agency",
                    "Contract Award Type": "Test Type",
                    "Funding Agency": "Test Funding Agency",
                    "Funding Sub Agency": "Test Sub Agency",
                    "Description": "Test Description",
                    "Last Modified Date": "2023-12-01",
                    "Base Obligation Date": "2023-12-01"
                }
            ],
            "page_metadata": {"hasNext": False},
        },
        status_code=200
    )

    # Mock the `requests.post` method
    mocker.patch('app.ml_app.requests.post', return_value=mock_api_response)

    # Mock Cassandra session to avoid database operations
    mock_cassandra_session = mocker.patch('app.ml_app.cassandra_session.execute', MagicMock())

    # Perform the POST request
    response = client.post(
        '/fetch_data',
        data={'frequency': 'daily', 'start_date': '2023-12-01', 'end_date': '2023-12-21'},
        follow_redirects=True
    )

    assert response.status_code == 200
    assert b"Fetched and inserted records into Cassandra" in response.data

    mock_logger.info.assert_any_call("Data fetching frequency: %s, Pages per batch: %d", "daily", 20)

    # Verify the mocked Cassandra session was called
    assert mock_cassandra_session.called


def test_cassandra_connection():
    contact_points = [config.CASSANDRA_HOST]
    port = config.CASSANDRA_PORT

    try:
        # Connect to Cassandra
        cluster = Cluster(contact_points=contact_points, port=port)
        cluster.connect()

        # Check cluster metadata as a quick health check
        metadata = cluster.metadata
        assert metadata is not None, "Failed to retrieve cluster metadata. Cassandra might not be running."
        cluster.shutdown()

    except Exception as e:
        pytest.fail(f"Unable to connect to Cassandra: {e}")
