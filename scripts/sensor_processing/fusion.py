# ****************************************************************************
# *  Script process thermal video, fisheye rgb video, and mmWave radar data to determine probability of obstacles in terms of angles.

import cv2
import numpy as np
import pandas as pd
import math

# ===================== DATA PATHS =====================
THERMAL_PATH = ""
FISHEYE_PATH = ""
MMWAVE_PATH = ""


# ===================== GENERAL =====================

def detect_horizon(frame, ransac_iters=200, inlier_thresh=1.5, confidence_thresh=0.5, fallback_row=0):
    """
    Uses RANSAC to detemine horizon (water-land or water-sky interface).
    
    Inputs:
        frame (np.ndarray): Input RGB image.
        inlear_thresh (float): Distance threshold to determine inliers for RANSAC algorithm.
        confidence_thresh (float): Confidence threshold to return found horizon or fallback_row.
        fallback_row (int): Default row to use in final mask if RANSAC edge does not pass confidence_thresh. Can incorporate IMU altitude measurements in future work.

    Returns:
        mask (np.ndarray): Binary mask where pixels above horizon are 0 and pixels below horizon are 255.
        slope (float): Slope of found horizon line.
        intercept (float): Intercept of found horizon line.
        confidence (float): Confidence level of found horizon line, depends on length of line.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    pts = np.column_stack(np.where(edges > 0))
    if len(pts) < 2:
        slope = 0
        intercept = fallback_row
        return slope, intercept, 0.0, np.ones_like(gray, dtype=np.uint8) * 255

    rows, cols = gray.shape
    best_inliers = []
    best_model = None

    # RANSAC iterations
    for _ in range(ransac_iters):
        idxs = np.random.choice(len(pts), 2, replace=False)
        (y1, x1), (y2, x2) = pts[idxs]
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        distances = np.abs(slope * pts[:, 1] - pts[:, 0] + intercept) / np.sqrt(slope ** 2 + 1)
        inliers = pts[distances < inlier_thresh]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (slope, intercept)

    if best_model is None:
        slope = 0
        intercept = fallback_row
        return slope, intercept, 0.0, np.ones_like(gray, dtype=np.uint8) * 255

    slope, intercept = best_model

    if len(best_inliers) > 0:
        x_span = best_inliers[:, 1].max() - best_inliers[:, 1].min()
        confidence = x_span / cols
    else:
        confidence = 0.0

    if confidence < confidence_thresh:
        slope = 0
        intercept = fallback_row

    mask = np.zeros_like(gray, dtype=np.uint8)
    for x in range(cols):
        y_line = int(slope * x + intercept)
        if y_line < 0:
            y_line = 0
        elif y_line >= rows:
            y_line = rows - 1
        mask[y_line:, x] = 255

    return mask, slope, intercept, confidence

def extract_kp(keypoints, cx, pix_deg_ratio):
    '''
    Extracts pixel coordinates, object sizes, and angles of attack from detected keypoints.

    Inputs:
        keypoints (cv2.KeyPoint):
        cx (int): Optical centre
        pix_deg_ratio (float): Ratio of pixels to degrees for image (experimentally determined).

    Returns:
        coords (list): List of x,y pixel coordinates of detected blobs.
        sizes (list): List of sizes of detected blobs.
        angles (list): List of angles of detected blobs.
    '''
    coords = []
    sizes = []
    angles = []
    
    for kp in keypoints:
        coords.append([ int(kp.pt[0]), int(kp.pt[1]) ])
        sizes.append(kp.size)
        angles.append((kp.pt[0] - cx)/ pix_deg_ratio)

    return coords, sizes, angles

def create_bins(bin_range, step):
    """
    Create bins based on a given range and step. Returns the bin labels as the middle value of each bin.
    """
    bins = np.arange(bin_range[0], bin_range[1] + step, step)
    bin_labels = [f"{(bins[i] + bins[i + 1]) / 2}°" for i in range(len(bins) - 1)]
    return bins, bin_labels

def assign_to_bin(value, bins):
    """
    Assign a value to a bin.
    """
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return None  # If the value is outside the defined bins


# ===================== THERMAL =====================

thermal_cap = cv2.VideoCapture(THERMAL_PATH)
if not thermal_cap.isOpened():
    print("Error: Cannot open thermal video.")

K_thermal=np.array([[63.78, 0.0, 80.44], [0.0, 64.12, 62.40], [0.0, 0.0, 1.0]])
D_thermal=np.array([-0.0446, -0.00027235, -0.0050242577, -0.003910302, 0.00086112796])

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 180
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 4
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
thermal_detector = cv2.SimpleBlobDetector_create(params)

average = 127.5

def undistort_thermal(frame, K, D):
    '''
    Undistorts thermal image using camera intrinsics.

    Inputs:
        frame (np.ndarray): Input RGB image.
        K (np.ndarray): Camera intrinsic matrix.
        D (np.ndarray): Camera distortion coefficients.

    Returns:
        undistorted (np.ndarray): Undistorted RGB image.    
    '''
    h,w = frame.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=0)
    undistorted = cv2.undistort(frame, K, D, None, new_K)

    return undistorted

def thermal_obstacle_detection(frame, average, mask, object_thresh):
    '''
    Detects obstacles in thermal camera images using a mix of normalizing and thresholding.

    Inputs:
        frame (np.ndarray): Input RGB image.
        average (float): Running average intensity of images.
        object_thresh (int): Threshold above which to extract objects.

    Returns:
        undistorted (np.ndarray): Undistorted RGB image.    
        new_average (float): Average of new input frame.
    '''
    h,w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_average = cv2.mean(gray)[0]

    sub = cv2.subtract(gray, average)
    maxVal = np.max(sub)

    if maxVal > 50: # prevent extremely low contrast scenes from being detected
        # increase contrast within the image
        normalized = (sub.astype(np.float32) / maxVal) * 255
        normalized = normalized.astype(np.uint8)
    else:
        normalized = sub

    normalized = cv2.bitwise_and(normalized, normalized, mask=mask)

    ret, obstacle_mask = cv2.threshold(normalized, object_thresh, 255, 0)
    
    return obstacle_mask, new_average


# ===================== FISHEYE =====================

fisheye_cap = cv2.VideoCapture(FISHEYE_PATH)
if not fisheye_cap.isOpened():
    print("Error: Cannot open fisheye video.")

K_fisheye=np.array([[418.51, 0.0, 444.11], [0.0, 418.58, 300.50], [0.0, 0.0, 1.0]])
D_fisheye=np.array([[-0.029317890988973864], [0.008321019929328282], [-0.022020233653477182], [0.010100750499314356]])

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 180
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 20
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
rgb_detector = cv2.SimpleBlobDetector_create(params)

def undistort_fisheye(frame, K, D):
    '''
    Undistorts fisheye image using camera intrinsics.

    Inputs:
        frame (np.ndarray): Input RGB image.
        K (np.ndarray): Camera intrinsic matrix.
        D (np.ndarray): Camera distortion coefficients.

    Returns:
        undistorted (np.ndarray): Undistorted RGB image.    
    '''
    h,w = frame.shape[:2]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted

def gradient_obstacle_detection(frame, gaussian_window=5, gradient_threshold=30):
    """
    Detect obstacles in water scenes using gradient filtering. Based on paper https://doi.org/10.1371/journal.pone.0205319.
    
    Inputs:
        frame (np.ndarray): Input BGR image.
        gaussian_window (int): Gaussian blur kernel size for gradient smoothing.
        
    Returns:
        obstacle_mask (np.ndarray): Binary mask of detected obstacle.
    """

    # Reduce glint in image by hsv thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    glint_mask = cv2.inRange(v, 240, 255)
    v_inpaint = cv2.inpaint(v, glint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    hsv_reduced = cv2.merge([h, s, v_inpaint])
    glint_reduced = cv2.cvtColor(hsv_reduced, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(glint_reduced, cv2.COLOR_BGR2GRAY)

    # Gradient-based segmentation
    blurred = cv2.GaussianBlur(gray, (gaussian_window, gaussian_window), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_x_abs = cv2.convertScaleAbs(grad_x)

    _, obstacle_mask = cv2.threshold(grad_x_abs, gradient_threshold, 255, cv2.THRESH_BINARY)

    obstacle_mask = cv2.medianBlur(obstacle_mask, 3)

    return obstacle_mask

def extract_kp(keypoints, cx, pix_deg_ratio):
    '''
    Extracts pixel coordinates, object sizes, and angles of attack from detected keypoints.

    Inputs:
        keypoints (cv2.KeyPoint):
        cx (int): Optical centre
        pix_deg_ratio (float): Ratio of pixels to degrees for image (experimentally determined).

    Returns:
        coords (list): List of x,y pixel coordinates of detected blobs.
        sizes (list): List of sizes of detected blobs.
        angles (list): List of angles of detected blobs.
    '''
    coords = []
    sizes = []
    angles = []
    
    for kp in keypoints:
        coords.append([ int(kp.pt[0]), int(kp.pt[1]) ])
        sizes.append(kp.size)
        angles.append((kp.pt[0] - cx)/ pix_deg_ratio)

    return coords, sizes, angles


# ===================== MMWAVE =====================

df = pd.read_csv(MMWAVE_PATH)
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['RoundedTime'] = df['Timestamp'].apply(lambda x: x.strftime("%H:%M:%S.%f")[:-5])

grouped = df.groupby('RoundedTime')
timestamps = sorted(grouped.groups.keys())
current_index = [0]

def project_mmwave(mmwave_pts, T_thermal, T_rgb):
    '''
    Projects 3D points onto image planes of fisheye and RGB images.

    Inputs:
        mmwave_pts (list): List of mmWave points to project into 2D space.
        T_thermal (np.ndarray): Translation matrix between mmWave antenna centre and thermal camera centre.
        T_rgb (np.ndarray): Translation matrix between mmWave antenna centre and RGB camera centre.

    Returns:
        thermal_pts (list): List of mmWave points as thermal image pixel coordinates.
        rgb_pts (list): List of mmWave points as RGB image pixel coordinates.
    '''
    thermal_pts = []
    rgb_pts = []
    for pt in mmwave_pts:
        X,Y,Z = pt
        thermal_x = T_thermal[0] * (X + T_thermal[1]) / (Y + T_thermal[2]) + T_thermal[3]
        rgb_x = T_rgb[0] * (X + T_rgb[1]) / (Y + T_rgb[2]) + T_rgb[3]
        thermal_pts.append([int(thermal_x), Y])
        rgb_pts.append([int(rgb_x), Y])

    return thermal_pts, rgb_pts

def mmwave_angles(mmwave_pts):
    '''
    Converts 3D points to azimuth angles.

    Inputs:
        mmwave_pts (list): List of mmWave points to convert.

    Returns:
        angles (list): List of azimuth angles of mmWave detected object points.
    '''

    angles = []

    for pt in mmwave_pts:
        X,Y,Z = pt
        angles.append(math.atan(X/Y) * 180 / np.pi)

# ===================== MAIN =====================

bin_range = [-55, 55]
step = 10
bins, bin_labels = create_bins(bin_range, step)
n_bins = len(bins) - 1

for timestamp in timestamps:
    ret, thermal_frame = thermal_cap.read()
    if not ret:
        break
    ret, fisheye_frame = fisheye_cap.read()
    if not ret:
        break
    group = grouped.get_group(timestamp)

    t_undistorted = undistort_thermal(thermal_frame, K_thermal, D_thermal)
    t_horizon_mask, t_slope, t_intercept, t_confidence = detect_horizon(t_undistorted)
    t_masked_image = cv2.bitwise_and(t_undistorted, t_undistorted, mask=t_horizon_mask)

    t_obstacle_mask, new_average = thermal_obstacle_detection(t_masked_image, average, 100)

    average = (average + new_average) / 2

    t_keypoints = thermal_detector.detect(t_obstacle_mask)

    t_coords = []
    t_sizes = []
    t_angles = []

    if t_keypoints:
        t_coords, t_sizes, t_angles = extract_kp(t_keypoints, cx=77, pix_deg_ratio=2.82)

    rgb_undistorted = undistort_fisheye(fisheye_frame, K_fisheye, D_fisheye)

    rgb_horizon_mask, rgb_slope, rgb_intercept, rgb_confidence = detect_horizon(rgb_undistorted)
    rgb_masked_image = cv2.bitwise_and(rgb_undistorted, rgb_undistorted, mask=rgb_horizon_mask)

    rgb_obstacle_mask = gradient_obstacle_detection(rgb_masked_image)

    rgb_keypoints = rgb_detector.detect(rgb_obstacle_mask)

    rgb_coords = []
    rgb_sizes = []
    rgb_angles = []

    if rgb_keypoints:
        rgb_coords, rgb_sizes, rgb_angles = extract_kp(rgb_keypoints, cx=472, pix_deg_ratio=7.2)

    mm_angles = mmwave_angles(group[['X','Y','Z']].values)
    mm_ranges = group[['Y']].values

    # Process angle data into obstacle confidence bins
    bin_values = [0.0] * n_bins
    bin_min_distances = [None] * n_bins

    # Process each bin
    for bin_idx in range(n_bins):
        # Confidence value
        hit_count = 0
        for angle_list in (t_angles, rgb_angles, mm_angles):
            if any(assign_to_bin(angle, bins) == bin_idx for angle in angle_list):
                hit_count += 1

        if hit_count == 1:
            bin_values[bin_idx] = 0.33
        elif hit_count == 2:
            bin_values[bin_idx] = 0.66
        elif hit_count == 3:
            bin_values[bin_idx] = 1.0

        # Minimum distance for mm_angles
        distances_in_bin = [dist for angle, dist in zip(mm_angles, mm_ranges)
                            if assign_to_bin(angle, bins) == bin_idx]
        if distances_in_bin:
            bin_min_distances[bin_idx] = min(distances_in_bin)

    # Print results
    spacing = 12
    print("".join(f"Bin {i+1:<{spacing-4}}" for i in range(len(bin_labels))))
    print("".join(f"{label:<{spacing}}" for label in bin_labels))
    print("".join(f"{value:<{spacing}.2f}" for value in bin_values))
    print("".join(f"{str(dist) if dist is not None else '-':<{spacing}}" for dist in bin_min_distances))