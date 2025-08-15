# ****************************************************************************
# *  Script to process thermal videos to detect obstacles on the water.

import cv2
import numpy as np

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

############################################################################################################################################

THERMAL_PATH = "data/Ducks/thermal_2025-07-15_05-36-18.mp4"

thermal_cap = cv2.VideoCapture(THERMAL_PATH)
if not thermal_cap.isOpened():
    print("Error: Cannot open thermal video.")

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
detector = cv2.SimpleBlobDetector_create(params)

average = 127.5

K_thermal=np.array([[63.78, 0.0, 80.44], [0.0, 64.12, 62.40], [0.0, 0.0, 1.0]])
D_thermal=np.array([-0.0446, -0.00027235, -0.0050242577, -0.003910302, 0.00086112796])

while True:
    ret, frame = thermal_cap.read()
    if not ret:
        break

    frame_undistorted = undistort_thermal(frame, K_thermal, D_thermal)

    horizon_mask, slope, intercept, confidence = detect_horizon(frame_undistorted)
    masked_image = cv2.bitwise_and(frame_undistorted, frame_undistorted, mask=horizon_mask)

    obstacle_mask, new_average = thermal_obstacle_detection(masked_image, average, 100)

    average = (average + new_average) / 2

    keypoints = detector.detect(obstacle_mask)

    out = frame_undistorted.copy()

    coords = []
    sizes = []
    angles = []

    if keypoints:
        coords, sizes, angles = extract_kp(keypoints, cx=77, pix_deg_ratio=2.82)
        for x,y in zip(coords):
            cv2.circle(out, x, y, 6, (0,0,0), -1)

    cv2.imshow("out", out)
    key = cv2.waitKey(0)
    if key == 27:
        break
    continue

thermal_cap.release()
cv2.destroyAllWindows()