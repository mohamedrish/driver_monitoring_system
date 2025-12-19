import numpy as np
import cv2

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def compute_EAR(landmarks_px, eye_indices):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_indices = [p1, p2, p3, p4, p5, p6]
    """
    p1, p2, p3, p4, p5, p6 = [landmarks_px[i] for i in eye_indices]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)

def compute_MAR(landmarks_px):
    """
    A simple MAR using MediaPipe landmarks:
    vertical: upper inner lip (13) to lower inner lip (14)
    horizontal: left mouth corner (78) to right mouth corner (308)
    """
    top = landmarks_px[13]
    bottom = landmarks_px[14]
    left = landmarks_px[78]
    right = landmarks_px[308]

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal

def rotationMatrixToEulerAngles(R):
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    """
    sy = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # pitch
        y = np.arctan2(-R[2, 0], sy)      # yaw
        z = np.arctan2(R[1, 0], R[0, 0])  # roll
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

def estimate_head_pose(landmarks_px, img_w, img_h):
    """
    Head pose from 2D-3D correspondences (solvePnP).
    Uses a generic 3D face model, good enough for attention direction.
    Returns pitch, yaw, roll (deg) or None if solvePnP fails.
    """
    # 2D image points from MediaPipe (common stable points)
    # nose tip (1), chin (152), left eye outer (33), right eye outer (263),
    # left mouth (61), right mouth (291)
    idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([landmarks_px[i] for i in idx], dtype=np.float64)

    # Generic 3D model points (approximate, in mm)
    model_points = np.array([
        (0.0,    0.0,    0.0),    # Nose tip
        (0.0,  -63.6,  -12.5),    # Chin
        (-43.3,  32.7,  -26.0),   # Left eye outer
        (43.3,   32.7,  -26.0),   # Right eye outer
        (-28.9, -28.9,  -24.1),   # Left mouth
        (28.9,  -28.9,  -24.1)    # Right mouth
    ])

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotationMatrixToEulerAngles(R)
    return pitch, yaw, roll