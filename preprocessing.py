import cv2
import numpy as np

def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    1. HÀM LỌC NHIỄU MEDIAN (Lọc trung vị)
    --------------------------------------
    - Mục đích: Loại bỏ nhiễu hạt muối tiêu (salt-and-pepper noise) thường xuất hiện trên ảnh vệ tinh VIIRS.
    - Nguyên lý: Duyệt qua từng pixel, lấy giá trị trung vị của các pixel xung quanh trong vùng kernel.
    - Tham số:
        + image (np.ndarray): Ma trận ảnh đầu vào (ảnh xám / cường độ sáng VIIRS).
        + kernel_size (int): Kích thước cửa sổ lọc (phải là số lẻ: 3, 5, 7...). Mặc định = 3.
    - Trả về:
        + np.ndarray: Ma trận ảnh xám đã được khử nhiễu.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel_size phải là số lẻ (3, 5, 7, ...)")
        
    filtered_img = cv2.medianBlur(image, kernel_size)
    return filtered_img


def calculate_difference(img_2025: np.ndarray, img_2015: np.ndarray) -> np.ndarray:
    """
    2. HÀM TÍNH SAI PHÂN ÁNH SÁNG BAN ĐÊM (2025 - 2015)
    --------------------------------------------------
    - Mục đích: Xác định sự gia tăng cường độ ánh sáng ban đêm trong giai đoạn 10 năm tại Bắc Ninh.
    - Nguyên lý: Lấy giá trị từng pixel ảnh năm 2025 trừ đi ảnh năm 2015.
                 Những vùng phát triển công nghiệp/đô thị mới sẽ có mức sáng gia tăng đáng kể.
    - Tham số:
        + img_2025 (np.ndarray): Ảnh Bắc Ninh năm 2025 (đã qua lọc nhiễu).
        + img_2015 (np.ndarray): Ảnh Bắc Ninh năm 2015 (đã qua lọc nhiễu).
    - Trả về:
        + np.ndarray: Ảnh sai phân thể hiện vùng tăng trưởng ánh sáng.
    """
    if img_2025.shape != img_2015.shape:
        raise ValueError("Hai ảnh đầu vào phải có cùng kích thước (shape).")

    # Ép kiểu sang float32 để tránh lỗi trượt/tràn số khi phép trừ ra kết quả âm
    img_2025_float = img_2025.astype(np.float32)
    img_2015_float = img_2015.astype(np.float32)

    # Tính chênh lệch (chỉ giữ lại phần tăng trưởng sáng > 0)
    diff = img_2025_float - img_2015_float
    diff = np.clip(diff, 0, 255) # Giới hạn khoảng giá trị [0, 255]

    return diff.astype(np.uint8)


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    3. HÀM PHÂN NGƯỠNG TỰ ĐỘNG OTSU (Tách nhị phân)
    -----------------------------------------------
    - Mục đích: Chuyển ảnh sai phân mức xám thành ảnh nhị phân (Đen/Trắng) phân tách rõ nét giữa 
                 vùng tăng trưởng đô thị/KCN (màu Trắng = 255) và vùng nền/nhiễu (màu Đen = 0).
    - Nguyên lý: Thuật toán Otsu tự động tính toán ngưỡng T (Threshold) tối ưu dựa trên histogram,
                 sao cho phương sai giữa 2 nhóm (nền và đối tượng) là lớn nhất.
    - Tham số:
        + image (np.ndarray): Ảnh sai phân đầu vào (mức xám 8-bit).
    - Trả về:
        + np.ndarray: Ảnh nhị phân (Binary Mask) chứa 2 giá trị 0 và 255.
    """
    # Cờ cv2.THRESH_OTSU sẽ tự động tìm ngưỡng thích hợp (tham số threshold=0 sẽ bị ghi đè)
    optimal_thresh, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"[LOG] Đã tính toán ngưỡng Otsu tối ưu thành công: T = {optimal_thresh}")
    return binary_mask

