"""GPS helpers: great-circle distance and nearest-point lookup. Coordinates in data/*.csv are APPROXIMATE sample values
for the demo; production would use official GTFS / OpenStreetMap data."""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def nearest(lat, lon, df):
    d = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    i = int(np.argmin(d))
    return df["name"].iloc[i], float(d[i])
