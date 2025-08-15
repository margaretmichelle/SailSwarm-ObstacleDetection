# ****************************************************************************
# *  Script to process fisheye RGB videos using gradient filtering to detect obstacles on the water.  

import cv2
import numpy as np

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

# def estimate_global_motion(prev_gray, curr_gray):
#     # Detect ORB features and descriptors
#     orb = cv2.ORB_create(500)
#     kp1, des1 = orb.detectAndCompute(prev_gray, None)
#     kp2, des2 = orb.detectAndCompute(curr_gray, None)
#     if des1 is None or des2 is None:
#         return None

#     # Match features with BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     if len(matches) < 10:
#         return None

#     matches = sorted(matches, key=lambda x: x.distance)

#     pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

#     return M

# def compensate_motion(frame, M):
#     # Warp frame with affine transform M to compensate global motion
#     h, w = frame.shape[:2]
#     compensated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
#     return compensated

# def sparse_optical_flow(prev_gray, curr_gray, prev_pts):
#     # Calculate sparse optical flow for tracked points
#     curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None,
#                                                      winSize=(15,15), maxLevel=2,
#                                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     return curr_pts, status

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

############################################################################################################################################

FISHEYE_PATH = "data/Rain/fisheye_2025-07-15_02-59-02.mp4"

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
detector = cv2.SimpleBlobDetector_create(params)

while True:
    ret, frame = fisheye_cap.read()
    if not ret:
        break

    frame_undistorted = undistort_fisheye(frame, K_fisheye, D_fisheye)

    horizon_mask, slope, intercept, confidence = detect_horizon(frame_undistorted)
    masked_image = cv2.bitwise_and(frame_undistorted, frame_undistorted, mask=horizon_mask)

    obstacle_mask = gradient_obstacle_detection(masked_image)

    keypoints = detector.detect(obstacle_mask)

    out = frame_undistorted.copy()

    coords = []
    sizes = []
    angles = []

    if keypoints:
        coords, sizes, angles = extract_kp(keypoints, cx=472, pix_deg_ratio=7.2)
        for x,y in zip(coords):
            cv2.circle(out, x, y, 6, (0,0,255), -1)

    cv2.imshow("out", out)
    key = cv2.waitKey(0)
    if key == 27:
        break
    continue

fisheye_cap.release()
cv2.destroyAllWindows()
