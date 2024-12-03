import json
from math import radians, sin, cos, sqrt, atan2
import os

# Haversine formula to calculate the distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function to find the closest ISO region
def find_closest_iso_region(coordinates):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    path = current_file_dir + "/iso_company.json"
    iso_company_map = load_iso_company_map(path)

    lat, lon = map(float, coordinates.split(","))
    closest_region = None
    min_distance = float("inf")
    
    for region, data in iso_company_map.items():
        region_lat, region_lon = map(float, data["coordinates"].split(","))
        distance = haversine(lat, lon, region_lat, region_lon)
        
        if distance < min_distance:
            min_distance = distance
            closest_region = region
    
    if closest_region:
        return iso_company_map[closest_region]["companies"]
    else:
        return []

# Load the iso_company_map from a JSON file
def load_iso_company_map(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

