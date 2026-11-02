"""Official GBR geometry, applied identically to all model features."""
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

BOUNDARY_FILE = Path(__file__).parent/"data"/"gbr_marine_park.geojson.gz"
BOUNDARY_SOURCE = "https://services-ap1.arcgis.com/8gXWSCxaJlFIfiTr/ArcGIS/rest/services/Great_Barrier_Reef_Marine_Park_Boundary_20/FeatureServer/63"


def boundary_identity():
    return {"source": BOUNDARY_SOURCE, "crs": "EPSG:4326", "license": "CC-BY-4.0",
            "sha256_geojson": hashlib.sha256(gzip.decompress(BOUNDARY_FILE.read_bytes())).hexdigest(),
            "rule": "cell_center_intersects_official_marine_park"}


@lru_cache(maxsize=1)
def geometry():
    from shapely.geometry import shape
    from shapely.ops import unary_union
    data = json.loads(gzip.decompress(BOUNDARY_FILE.read_bytes()))
    return unary_union([shape(feature["geometry"]) for feature in data["features"]])


def region_mask(latitude, longitude):
    lon, lat = np.meshgrid(longitude, latitude)
    return contains_points(lat, lon)


def contains_points(latitude, longitude):
    from shapely import intersects_xy
    return intersects_xy(geometry(), np.asarray(longitude), np.asarray(latitude))
