import json
import cv2
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom

def apply_median_filter(image, kernel_size=3):
    """Lấy trung vị các pixel lân cận để giảm nhiễu; trả ảnh cùng kích thước."""
    image = np.ma.asarray(image, dtype=np.float32).filled(np.nan)
    if image.ndim != 2 or not isinstance(kernel_size, (int, np.integer)) or kernel_size not in (3, 5):
        raise ValueError("Cần ảnh 2D và kernel 3 hoặc 5 cho ảnh bức xạ.")
    valid = np.isfinite(image)
    if not valid.any():
        raise ValueError("Ảnh không có dữ liệu hợp lệ.")
    filled = np.where(valid, image, 0).astype(np.float32)
    result = cv2.medianBlur(filled, int(kernel_size))

    if not valid.all():
        image = np.where(valid, image, np.nan)
        padded = np.pad(image, kernel_size // 2, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
        near_nodata = cv2.dilate((~valid).astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) > 0
        selected = near_nodata & valid
        result[selected] = np.nanmedian(windows[selected], axis=(-2, -1))
        result[~valid] = np.nan
    return result

def calculate_difference(img_2025, img_2015):
    """2025 − 2015: dương là sáng lên, âm là tối đi."""
    current = np.ma.asarray(img_2025, dtype=np.float64).filled(np.nan)
    baseline = np.ma.asarray(img_2015, dtype=np.float64).filled(np.nan)
    if current.ndim != 2 or current.shape != baseline.shape:
        raise ValueError("Hai ảnh phải cùng kích thước 2D.")
    difference = current - baseline
    difference[~(np.isfinite(current) & np.isfinite(baseline))] = np.nan
    return difference

def apply_otsu_threshold(image):
    """Otsu tự chọn ngưỡng; trả mask đen=0, trắng=255, NoData=0."""
    image = np.ma.asarray(image, dtype=np.float64).filled(np.nan)
    valid = np.isfinite(image)
    if image.ndim != 2 or not valid.any():
        raise ValueError("Cần ảnh 2D có dữ liệu hợp lệ.")
    values = image[valid]
    low, high = values.min(), values.max()
    if high == low:
        encoded = np.full(values.shape, 255 if high > 0 else 0, dtype=np.uint8)
    else:
        encoded = np.rint((values - low) / (high - low) * 255).astype(np.uint8)
    _, labels = cv2.threshold(encoded.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = np.zeros(image.shape, dtype=np.uint8)
    binary[valid] = labels.ravel()
    return binary

def apply_spatial_masking(raster_path, geojson_path, geojson_crs="EPSG:4326"):
    """Đọc ảnh VIIRS và GeoJSON Bắc Ninh; trả ảnh đã cắt, ngoài tỉnh là NaN."""
    with open(geojson_path, encoding="utf-8-sig") as file:
        boundary = json.load(file)
    if boundary["type"] == "FeatureCollection":
        shapes = [feature["geometry"] for feature in boundary["features"]]
    elif boundary["type"] == "Feature":
        shapes = [boundary["geometry"]]
    else:
        shapes = [boundary]

    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError("Ảnh cần có hệ tọa độ.")
        shapes = [transform_geom(geojson_crs, source.crs, shape) for shape in shapes]
        clipped, _ = mask(source, shapes, crop=True, filled=False, indexes=1)
        image = clipped.astype(np.float64).filled(np.nan)
        return image * source.scales[0] + source.offsets[0]

def calculate_economic_metrics(binary_mask, original_image, sol_2015=None):
    """Dùng mask chọn pixel, lấy bức xạ gốc để tính 4 chỉ số theo đề."""
    image = np.ma.asarray(original_image, dtype=np.float64).filled(np.nan)
    binary = np.asarray(binary_mask)
    if image.ndim != 2 or binary.shape != image.shape:
        raise ValueError("Mask và ảnh phải cùng kích thước 2D.")
    if not np.isfinite(image).any() or not np.isin(binary, [0, 1, 255]).all():
        raise ValueError("Cần ảnh hợp lệ và mask 0/1 hoặc 0/255.")
    urban = (binary > 0) & np.isfinite(image)
    if (image[urban] < 0).any():
        raise ValueError("Mask đang chọn bức xạ âm; cần kiểm tra dữ liệu.")

    area = int(np.count_nonzero(urban)) * 0.25
    sol = float(np.sum(image[urban]))
    density = sol / area if area > 0 else None
    if sol_2015 is not None and (not np.isfinite(sol_2015) or sol_2015 < 0):
        raise ValueError("SOL năm 2015 phải hữu hạn và không âm.")
    growth = (sol - sol_2015) / sol_2015 * 100 if sol_2015 is not None and sol_2015 > 0 else None

    print(f"1. Diện tích phát sáng: {area:.2f} km²")
    print(f"2. Tổng bức xạ (SOL): {sol:.4f}")
    print("3. Mật độ (SOL/km²):", f"{density:.4f}" if density is not None else "N/A")
    print("4. Tăng trưởng SOL so với 2015:", f"{growth:.2f}%" if growth is not None else "N/A")
    return {"area_km2": area, "sol": sol, "density": density, "growth_percent": growth}