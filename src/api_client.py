"""
weather_api_client.py

Module for interacting with the data.gov.sg v2 weather API. This client provides methods
to fetch and store both real-time and historical weather data in CSV format. Real-time
data is downloaded daily per sensor type, while historical data is retrieved by walking
through dataset resources exposed by the API.

This module handles pagination, data flattening, error logging, and file persistence,
ensuring resilience and traceability across data ingestion workflows.
"""

# Future Imports (to ensure compatibility with future Python versions)
from __future__ import annotations

# Standard Library Imports
import csv
import logging
import logging.config
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Third-Party Libraries
import requests
import yaml

# Pandas Import
import pandas as pd

###############################################################################
# Configure module-level logger
###############################################################################
logger = logging.getLogger(__name__)


class WeatherAPIClient:
    """
    Client for downloading real-time and historical weather data from
    data.gov.sg's v2 API endpoints.
    """

    # ────────────────────────────────
    # Endpoint configuration
    # ────────────────────────────────
    REALTIME_BASE_URL: str = "https://api-open.data.gov.sg/v2/real-time/api"

    API_BASE = "https://api-open.data.gov.sg/v2/real-time/api"
    FEATURES = ["rainfall", "wind-speed", "wind-direction", "air-temperature", "relative-humidity"]


    HISTORICAL_METADATA_ENDPOINTS: Dict[str, str] = {
        "rainfall": "https://api-production.data.gov.sg/v2/public/api/collections/2279/metadata",
        "air_temperature": "https://api-production.data.gov.sg/v2/public/api/collections/2246/metadata",
        "wind_speed": "https://api-production.data.gov.sg/v2/public/api/collections/2280/metadata",
        "wind_direction": "https://api-production.data.gov.sg/v2/public/api/collections/2281/metadata",
        "relative_humidity": "https://api-production.data.gov.sg/v2/public/api/collections/2278/metadata",
    }

    HISTORICAL_DATASTORE_ENDPOINT: str = "https://data.gov.sg/api/action/datastore_search"
    HISTORICAL_PAGE_LIMIT: int = 10_000  # Maximum allowed page size

    def __init__(
        self,
        realtime_dir: str = "./data",
        historical_dir: str = "./historical",
    ) -> None:
        """
        Initialize the client and ensure the data directories exist.

        Args:
            realtime_dir: Path to store real-time sensor data.
            historical_dir: Path to store historical datasets.
        """
        self.realtime_dir = realtime_dir
        self.historical_dir = historical_dir
        os.makedirs(realtime_dir, exist_ok=True)
        os.makedirs(historical_dir, exist_ok=True)

        logger.debug(
            "WeatherAPIClient initialised (realtime_dir=%s, historical_dir=%s)",
            realtime_dir,
            historical_dir,
        )

    def fetch_realtime_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Download real-time sensor readings from start_date to end_date (inclusive),
        traversing backwards to avoid overwriting partial data.

        Args:
            start_date: The most recent date to start fetching.
            end_date: The oldest date to stop fetching (inclusive).
        """
        sensors: Dict[str, str] = {
            "rainfall": "rainfall",
            "air_temperature": "air-temperature",
            "relative_humidity": "relative-humidity",
            "wind_direction": "wind-direction",
            "wind_speed": "wind-speed",
        }

        current_date = start_date
        while current_date >= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info("Fetching real-time data for %s", date_str)

            for sensor_name, endpoint in sensors.items():
                try:
                    readings = self._fetch_realtime_pages(endpoint, date_str)
                    rows = self._parse_realtime_readings(readings)
                    self._append_rows_to_csv(sensor_name, rows, directory=self.realtime_dir)
                    logger.info("%s: %d readings written", sensor_name, len(rows))
                except Exception as exc:
                    logger.error(
                        "Error fetching %s on %s: %s",
                        sensor_name,
                        date_str,
                        exc,
                        exc_info=True,
                    )

            current_date -= timedelta(days=1)

    def _fetch_realtime_pages(
        self,
        endpoint: str,
        date_str: str,
    ) -> List[Dict]:
        """
        Fetch paginated real-time readings for a single sensor on a specific date.

        Args:
            endpoint: API endpoint for the sensor.
            date_str: Date string (YYYY-MM-DD) for which to fetch data.

        Returns:
            A list of reading dictionaries from the API.
        """
        url = f"{self.REALTIME_BASE_URL}/{endpoint}"
        all_readings: List[Dict] = []
        pagination_token: Optional[str] = None

        while True:
            params: Dict[str, str] = {"date": date_str}
            if pagination_token:
                params["paginationToken"] = pagination_token

            logger.debug("GET %s params=%s", url, params)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            result = response.json().get("data", {})
            all_readings.extend(result.get("readings", []))
            pagination_token = result.get("paginationToken")

            if not pagination_token:
                break

        logger.debug(
            "Fetched %d total readings for endpoint=%s date=%s",
            len(all_readings),
            endpoint,
            date_str,
        )
        return all_readings

    @staticmethod
    def _parse_realtime_readings(readings: List[Dict]) -> List[Tuple[str, str, float]]:
        """
        Flatten real-time JSON readings into (timestamp, station_id, value) tuples.

        Args:
            readings: Raw nested readings from the API.

        Returns:
            A list of (timestamp, station_id, value) tuples.
        """
        rows: List[Tuple[str, str, float]] = []
        for reading in readings:
            timestamp = reading.get("timestamp")
            for entry in reading.get("data", []):
                station_id = entry.get("stationId")
                value = entry.get("value")
                rows.append((timestamp, station_id, value))
        return rows

    def fetch_historical_data(self) -> None:
        """
        Download and store all historical datasets available for each configured sensor.
        """
        for sensor_name, metadata_url in self.HISTORICAL_METADATA_ENDPOINTS.items():
            logger.info("Processing historical data for %s", sensor_name)
            try:
                resource_ids = self._get_child_dataset_ids(metadata_url)
                for resource_id in resource_ids:
                    logger.info("↪ Fetching dataset: %s", resource_id)
                    records = self._fetch_all_records(resource_id)
                    self._save_records_to_csv(sensor_name, resource_id, records)
            except Exception as exc:
                logger.error("Error processing %s: %s", sensor_name, exc, exc_info=True)

    @staticmethod
    def _get_child_dataset_ids(metadata_url: str) -> List[str]:
        """
        Extract child dataset resource IDs from the metadata endpoint.

        Args:
            metadata_url: Metadata endpoint for a collection.

        Returns:
            A list of resource IDs (strings).
        """
        logger.debug("GET %s", metadata_url)
        resp = requests.get(metadata_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        child_ids = data["data"]["collectionMetadata"]["childDatasets"]
        logger.debug("Found %d child datasets", len(child_ids))
        return child_ids

    def _fetch_all_records(self, resource_id: str) -> List[Dict]:
        """
        Fetch all records for a given dataset resource via pagination.

        Args:
            resource_id: The dataset resource identifier.

        Returns:
            List of record dictionaries.
        """
        all_records: List[Dict] = []
        offset = 0
        limit = self.HISTORICAL_PAGE_LIMIT

        while True:
            params: Dict[str, Union[int, str]] = {
                "resource_id": resource_id,
                "limit": limit,
                "offset": offset,
            }

            logger.debug(
                "GET %s params=%s (page_offset=%d)",
                self.HISTORICAL_DATASTORE_ENDPOINT,
                params,
                offset,
            )
            resp = requests.get(self.HISTORICAL_DATASTORE_ENDPOINT, params=params, timeout=60)
            resp.raise_for_status()
            result = resp.json()["result"]

            records = result.get("records", [])
            if not records:
                logger.debug("No more records returned; stopping pagination.")
                break

            all_records.extend(records)
            logger.debug("Accumulated %d records so far", len(all_records))

            if "_links" in result and "next" in result["_links"]:
                offset += limit
            else:
                break

        logger.info("Total records fetched: %d", len(all_records))
        return all_records

    def _save_records_to_csv(
        self,
        sensor_name: str,
        resource_id: str,
        records: List[Dict],
    ) -> None:
        """
        Save a full dataset to disk in CSV format.

        Args:
            sensor_name: Name of the sensor the dataset relates to.
            resource_id: Unique resource identifier of the dataset.
            records: List of record dictionaries to write.
        """
        if not records:
            logger.warning("No data found for dataset %s – nothing written", resource_id)
            return

        csv_path = os.path.join(self.historical_dir, f"{sensor_name}_{resource_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        logger.info("Saved %d records to %s", len(records), csv_path)

    def _append_rows_to_csv(
        self,
        sensor_name: str,
        rows: List[Tuple[str, str, float]],
        directory: str,
    ) -> None:
        """
        Append real-time readings to a CSV file for a given sensor.

        Args:
            sensor_name: Name of the sensor (e.g., "rainfall").
            rows: List of rows in (timestamp, station_id, value) format.
            directory: Path to the directory in which to write the file.
        """
        if not rows:
            logger.debug("No rows to append for sensor=%s", sensor_name)
            return

        csv_path = os.path.join(directory, f"{sensor_name}.csv")
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(["timestamp", "station_id", sensor_name])
            writer.writerows(rows)

        logger.debug("Appended %d rows to %s", len(rows), csv_path)


    def configure_logging(self, config_path: str = "logging.yaml") -> None:
        """
        Load a YAML-based logging configuration, falling back to a basic setup on failure.

        Args:
            config_path: Path to the YAML logging config file.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                config_dict = yaml.safe_load(fp)
            logging.config.dictConfig(config_dict)
            logger.info("Loaded logging configuration from %s", config_path)
        except Exception as exc:
            logging.basicConfig(level=logging.INFO)
            logger.warning(
                "Failed to load logging.yaml (%s) - falling back to basicConfig.", exc
            )

    def fetch_feature_history(self, feature: str, date: datetime):
        """
        Fetches feature readings for `date` and the day before.
        Handles pagination and returns a DataFrame with datetime index and one column of numeric values.
        """
        dfs = []
        for i in range(4, -1, -1):  # 4 days ago → today
            dt = date - timedelta(days=i)
            d_str = dt.strftime("%Y-%m-%d")
            logger.info(f"Fetching {feature} data for {d_str}")
            url = f"{self.API_BASE}/{feature}?date={d_str}"
            page = 1

            while url:
                logger.debug(f"→ Requesting page {page} for {feature} on {d_str}")
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"Request failed for {url}: {e}", exc_info=True)
                    break

                js = resp.json()
                items = js.get("data", {}).get("readings", [])
                if not items:
                    logger.warning(f"No readings found for {feature} on {d_str} (page {page})")
                    break

                for slice in items:
                    ts = pd.to_datetime(slice["timestamp"])
                    for rec in slice["data"]:
                        dfs.append({"ts": ts, "value": rec["value"]})

                next_token = js.get("paginationToken")
                if next_token:
                    url = f"{self.API_BASE}/{feature}?date={d_str}&paginationToken={next_token}"
                    page += 1
                else:
                    break

            logger.info(f"{feature} on {d_str}: {len(dfs)} readings collected (across {page} page{'s' if page > 1 else ''})")

        df = pd.DataFrame(dfs).set_index("ts").sort_index()
        return df



