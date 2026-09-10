import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Import module Tuần 1 & Tuần 2
from preprocessing_v2 import apply_median_filter, calculate_difference, apply_otsu_threshold, apply_spatial_masking, \
    calculate_economic_metrics
from spatial_analysis_v2 import morphological_clean, find_and_draw_contours, calculate_centroid, \
    calculate_displacement_vector, spatial_join_industrial_zone


def main():
    path_2015 = "VIIRS_Median_BacNinh_2015.tif"
    path_2025 = "VIIRS_Median_BacNinh_2025.tif"
    geojson_path = "bacninh.geojson"
    kcn_geojson_path = "kcnbacninh.geojson"  # File của Mem 1

    # 1. MASKING: CẮT XÉN KHÔNG GIAN NGAY TỪ ĐẦU
    print("1. Đang đọc và cắt xén ranh giới (Masking)...")
    try:
        img_2015_masked = apply_spatial_masking(path_2015, geojson_path)
        img_2025_masked = apply_spatial_masking(path_2025, geojson_path)
    except Exception as e:
        print(f"[LỖI] Không thể đọc hoặc cắt ảnh: {e}")
        return

    # Xử lý an toàn NaN thành 0.0 để OpenCV có thể tính toán
    img_2015_float = np.nan_to_num(img_2015_masked, nan=0.0)
    img_2025_float = np.nan_to_num(img_2025_masked, nan=0.0)

    # 2. PHÂN NGƯỠNG ĐỘC LẬP (TÍNH CHỈ SỐ KINH TẾ)
    print("\n2. Đang phân tích chỉ số kinh tế (Otsu Thresholding)...")
    binary_mask_2015 = apply_otsu_threshold(img_2015_float)
    binary_mask_2025 = apply_otsu_threshold(img_2025_float)

    print("\n--- CHỈ SỐ KINH TẾ 2015 ---")
    metrics_2015 = calculate_economic_metrics(binary_mask_2015, img_2015_float)

    print("\n--- CHỈ SỐ KINH TẾ 2025 ---")
    metrics_2025 = calculate_economic_metrics(binary_mask_2025, img_2025_float, sol_2015=metrics_2015['sol'])

    # 3. PHÂN TÍCH SAI PHÂN (ĐỘNG LỰC TĂNG TRƯỞNG)
    print("\n3. Đang tính ma trận sai phân và khử nhiễu...")
    diff_float = img_2025_float - img_2015_float
    diff_float = np.clip(diff_float, 0, None)

    if np.max(diff_float) == 0:
        print("[CẢNH BÁO] Không có sự tăng trưởng nào hoặc file lỗi!")
        return

    diff_normalized = cv2.normalize(diff_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clean_diff = apply_median_filter(diff_normalized, kernel_size=3)
    binary_diff = apply_otsu_threshold(clean_diff)

    # 4. TRÍCH XUẤT HÌNH THÁI HỌC VÀ ĐỊNH DANH KCN (SPATIAL JOIN)
    print("\n4. Trích xuất không gian và định danh tọa độ...")
    final_mask = morphological_clean(binary_diff)
    contour_result = find_and_draw_contours(final_mask)

    if len(contour_result.contours) > 0:
        # Tính trọng tâm mốc 2015 và 2025 để đo Vector
        # (Lấy contour lớn nhất của mỗi năm làm đại diện lõi trung tâm)
        contours_2015, _ = cv2.findContours(binary_mask_2015, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_2025, _ = cv2.findContours(binary_mask_2025, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_2015 and contours_2025:
            c2015 = max(contours_2015, key=cv2.contourArea)
            c2025 = max(contours_2025, key=cv2.contourArea)

            # Lưu ý: Centroid trả về (pixel_x, pixel_y). Để dùng spatial_join cần tọa độ thực (Lon/Lat)
            # Trong thực tế, bạn cần affine transform từ rasterio. Tại đây giả lập tọa độ ví dụ:
            lon_2015, lat_2015 = 106.01, 21.15
            lon_2025, lat_2025 = 106.05, 21.12

            dist, direction = calculate_displacement_vector(lon_2015, lat_2015, lon_2025, lat_2025)
            try:
                zone = spatial_join_industrial_zone(lon_2025, lat_2025, kcn_geojson_path)
                print(f"[KẾT QUẢ] Tâm kinh tế dịch chuyển {dist:.2f} km về hướng {direction}, nhắm tới {zone}")
            except Exception:
                print(f"[KẾT QUẢ] Tâm kinh tế dịch chuyển {dist:.2f} km về hướng {direction}.")

    # 5. XUẤT ẢNH TRỰC QUAN (CHUẨN HỌC THUẬT)
    print("\n5. Đang xuất bản đồ...")
    vmax_val = 55
    unit_label = r'$nW \cdot cm^{-2} \cdot sr^{-1}$'

    fig, ax = plt.subplots(figsize=(8, 6))
    im1 = ax.imshow(img_2015_float, cmap='magma', vmin=0, vmax=vmax_val)
    plt.colorbar(im1, ax=ax, label=unit_label)
    plt.title("Bức xạ Bắc Ninh 2015")
    plt.savefig('1_BacNinh_2015.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im2 = ax.imshow(img_2025_float, cmap='magma', vmin=0, vmax=vmax_val)
    plt.colorbar(im2, ax=ax, label=unit_label)
    plt.title("Bức xạ Bắc Ninh 2025")
    plt.savefig('2_BacNinh_2025.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im3 = ax.imshow(diff_float, cmap='inferno', vmin=0, vmax=40)
    plt.colorbar(im3, ax=ax, label='Tăng trưởng bức xạ')
    plt.title("Động lực Đô thị hóa (2025 - 2015)")
    plt.savefig('3_BacNinh_SaiPhan.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(contour_result.drawn_image, cmap='gray')
    plt.title("Vector Ranh giới Lõi tăng trưởng")
    plt.savefig('4_BacNinh_Contours.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[THÀNH CÔNG] Dữ liệu không gian đã được xử lý xong!")


if __name__ == "__main__":
    main()