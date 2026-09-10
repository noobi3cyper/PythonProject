# -*- coding: utf-8 -*-
"""Spatial analysis functions for the VIIRS-based urban growth study (Week 1 + Week 2)."""

import math
from pathlib import Path
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


# ==========================================
# CÁC HÀM TUẦN 1: HÌNH THÁI HỌC VÀ TRỌNG TÂM
# ==========================================

class ContourResult(list):
    """Lưu trữ kết quả đường viền và ảnh đã vẽ đường viền."""

    def __init__(self, contours, contour_image):
        super().__init__(contours)
        self.contours = list(contours)
        self.drawn_image = contour_image


def morphological_clean(binary_mask):
    """Làm sạch ảnh nhị phân bằng phép toán Mở (Opening) và Đóng (Closing)."""
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[-1] in (3, 4):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY if mask.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY)
    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def find_and_draw_contours(binary_mask):
    """Tìm đa giác lõi sáng và vẽ viền trực quan."""
    binary = np.asarray(binary_mask > 0, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rendered = np.zeros_like(binary)
    rendered = cv2.drawContours(rendered, contours, -1, 255, 1)
    return ContourResult(contours, rendered)


def calculate_centroid(contour):
    """Tính toán tọa độ pixel Trọng tâm bằng Moment không gian."""
    contour_array = np.asarray(contour, dtype=np.float32)
    if contour_array.size == 0:
        return np.array([0.0, 0.0], dtype=float)
    moments = cv2.moments(contour_array)
    m00 = moments.get("m00", 0.0)
    if m00 == 0:
        return np.array([0.0, 0.0], dtype=float)
    return np.array([moments["m10"] / m00, moments["m01"] / m00], dtype=float)


# ==========================================
# CÁC HÀM TUẦN 2: VECTOR VÀ SPATIAL JOIN
# ==========================================

def calculate_displacement_vector(lon_2015, lat_2015, lon_2025, lat_2025):
    """Tính khoảng cách (km) và hướng dịch chuyển từ tâm 2015 -> 2025."""
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat_2015), math.radians(lat_2025)
    dphi = math.radians(lat_2025 - lat_2015)
    dlambda = math.radians(lon_2025 - lon_2015)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = earth_radius_km * c

    dx, dy = lon_2025 - lon_2015, lat_2025 - lat_2015

    if dx == 0 and dy == 0:
        direction = "không dịch chuyển"
    elif dx == 0:
        direction = "Bắc" if dy > 0 else "Nam"
    elif dy == 0:
        direction = "Đông" if dx > 0 else "Tây"
    else:
        east_west = "Đông" if dx > 0 else "Tây"
        north_south = "Bắc" if dy > 0 else "Nam"
        direction = f"{east_west} {north_south}"

    return distance_km, direction


def _pick_zone_name(row):
    """Hỗ trợ lọc tên KCN từ các cột bị đặt tên lộn xộn trong GeoJSON."""
    preferred_columns = ["kcn_name", "KCN", "name", "Name", "ten", "Ten", "TEN"]
    for column in preferred_columns:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return None


def spatial_join_industrial_zone(lon, lat, geojson_path):
    """Tìm tên đa giác KCN chứa điểm (lon, lat) bằng Geopandas."""
    zones = gpd.read_file(geojson_path)
    if zones.crs is None:
        zones = zones.set_crs("EPSG:4326")
    else:
        zones = zones.to_crs("EPSG:4326")

    point_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    joined = gpd.sjoin(point_gdf, zones, how="left", predicate="within")

    if joined.empty:
        return "không xác định"

    row = joined.iloc[0]
    if pd.isna(row.get("index_right")):
        return "không xác định"

    zone_name = _pick_zone_name(row)
    if zone_name:
        return zone_name
    return f"feature_{int(row['index_right'])}"