import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd

# Import các hàm đã viết từ 2 file module
from preprocessing import apply_median_filter, calculate_difference, apply_otsu_threshold
from spatial_analysis import morphological_clean, find_and_draw_contours, calculate_centroid


def main():
    # 1. ĐỌC DỮ LIỆU TIF BẰNG RASTERIO
    path_2015 = "VIIRS_Median_BacNinh_2015.tif"
    path_2025 = "VIIRS_Median_BacNinh_2025.tif"

    try:
        with rasterio.open(path_2015) as src_2015:
            img_2015_float = src_2015.read(1).astype(np.float32)
            crs_2015 = src_2015.crs
            transform_2015 = src_2015.transform

        with rasterio.open(path_2025) as src_2025:
            img_2025_float = src_2025.read(1).astype(np.float32)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file .tif!")
        return

    img_2015_float = np.nan_to_num(img_2015_float, nan=0.0)
    img_2025_float = np.nan_to_num(img_2025_float, nan=0.0)

    # 2. TÍNH SAI PHÂN & CHUẨN HÓA
    print("Đang tính sai phân...")
    diff_float = img_2025_float - img_2015_float
    diff_float = np.clip(diff_float, 0, None)

    diff_normalized = cv2.normalize(diff_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. CHẠY PIPELINE THUẬT TOÁN
    print("Đang chạy thuật toán lọc nhiễu và phân ngưỡng...")
    clean_diff = apply_median_filter(diff_normalized, kernel_size=3)
    binary_mask = apply_otsu_threshold(clean_diff)

    final_mask = morphological_clean(binary_mask)
    contour_result = find_and_draw_contours(final_mask)

    # 4. CHUẨN BỊ BẢN ĐỒ RANH GIỚI
    try:
        boundary = gpd.read_file("bacninh.geojson")
        boundary = boundary.to_crs(crs_2015)
    except Exception as e:
        print(f"Lỗi đọc file GeoJSON: {e}")
        return

    # 5. XUẤT 4 ẢNH ĐỘC LẬP
    print("\nĐang xuất 4 file ảnh bản đồ...")
    vmax_val = 55

    # --- Ảnh 1: 2015 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    show(img_2015_float, transform=transform_2015, ax=ax, cmap='magma', vmin=0, vmax=vmax_val)
    boundary.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=2)
    plt.colorbar(ax.images[0], ax=ax, label='Radiance')
    plt.title("VIIRS DNB 2015 - Tỉnh Bắc Ninh")
    plt.savefig('1_BacNinh_2015_Boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Ảnh 2: 2025 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    show(img_2025_float, transform=transform_2015, ax=ax, cmap='magma', vmin=0, vmax=vmax_val)
    boundary.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=2)
    plt.colorbar(ax.images[0], ax=ax, label='Radiance')
    plt.title("VIIRS DNB 2025 - Tỉnh Bắc Ninh")
    plt.savefig('2_BacNinh_2025_Boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Ảnh 3: Sai phân ---
    fig, ax = plt.subplots(figsize=(10, 8))
    show(diff_float, transform=transform_2015, ax=ax, cmap='inferno', vmin=0, vmax=40)
    boundary.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=2)
    plt.colorbar(ax.images[0], ax=ax, label='Radiance Increase')
    plt.title("Gia tăng ánh sáng đô thị (2025 - 2015)")
    plt.savefig('3_BacNinh_SaiPhan_Boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Ảnh 4: Vector khoanh vùng ---
    fig, ax = plt.subplots(figsize=(10, 8))
    show(contour_result.drawn_image, transform=transform_2015, ax=ax, cmap='gray')
    boundary.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=2)
    plt.title("Thuật toán Trích xuất: Động lực tăng trưởng lõi")
    plt.savefig('4_BacNinh_Contours_Boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[THÀNH CÔNG] Đã lưu 4 file PNG có ranh giới vào thư mục dự án!")


if __name__ == "__main__":
    main()