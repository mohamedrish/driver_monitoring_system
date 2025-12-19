import numpy as np
from utils import compute_EAR, compute_MAR, estimate_head_pose

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

class DMSState:
    def __init__(self):
        self.closed_eye_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.phone_counter = 0

        # --- NEW: head pose calibration
        self.pose_bias_yaw = 0.0
        self.pose_bias_pitch = 0.0
        self.calibrated = False
        self.calib_samples = 0
        self.calib_sum_yaw = 0.0
        self.calib_sum_pitch = 0.0

        # --- NEW: smoothing (optional but recommended)
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0

        self.active_alerts = set()

def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def compute_events(cfg, state, landmarks_px, head_pose, yolo_result, face_box):
    """
    Returns:
      metrics dict + detected_events set
    """
    events = set()
    metrics = {}

    # EAR (drowsiness)
    left_ear = compute_EAR(landmarks_px, LEFT_EYE_IDX)
    right_ear = compute_EAR(landmarks_px, RIGHT_EYE_IDX)
    ear = (left_ear + right_ear) / 2.0
    metrics["EAR"] = ear

    # MAR (yawn)
    mar = compute_MAR(landmarks_px)
    metrics["MAR"] = mar

    # Head pose (distraction)
    if head_pose is not None:
       pitch, yaw, roll = head_pose

       # subtract neutral bias if calibrated
       if getattr(state, "calibrated", False):
          yaw = yaw - state.pose_bias_yaw
          pitch = pitch - state.pose_bias_pitch

    # smoothing (EMA)
       alpha = 0.2
       state.smooth_yaw = alpha * yaw + (1 - alpha) * state.smooth_yaw
       state.smooth_pitch = alpha * pitch + (1 - alpha) * state.smooth_pitch

       yaw_s = state.smooth_yaw
       pitch_s = state.smooth_pitch

       metrics["pitch"] = pitch_s
       metrics["yaw"] = yaw_s
       metrics["roll"] = roll

       if abs(yaw_s) > cfg.yaw_threshold_deg or abs(pitch_s) > cfg.pitch_threshold_deg:
          events.add("DISTRACTION")
          
    # Drowsiness event (based on threshold; temporal logic handled outside)
    if ear < cfg.ear_threshold:
        events.add("EYES_CLOSED")

    if mar > cfg.mar_threshold:
        events.add("YAWN")

    # Phone detection (YOLO)
    phone_detected = False
    phone_boxes = []
    if yolo_result is not None:
        names = yolo_result.names
        boxes = yolo_result.boxes
        for b in boxes:
            cls_id = int(b.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name == cfg.phone_class_name:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                phone_boxes.append((x1, y1, x2, y2))
                phone_detected = True

    metrics["phone_boxes"] = phone_boxes

    # Optional: make phone event stronger by requiring proximity to face
    if phone_detected:
        if face_box is None:
            events.add("PHONE")
        else:
            # if phone bbox intersects face bbox (or near it)
            fx1, fy1, fx2, fy2 = face_box
            for (x1, y1, x2, y2) in phone_boxes:
                inter_x1 = max(fx1, x1)
                inter_y1 = max(fy1, y1)
                inter_x2 = min(fx2, x2)
                inter_y2 = min(fy2, y2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    events.add("PHONE")
                    break
            else:
                # still flag, but you can choose to not alert
                events.add("PHONE")

    return metrics, events

def update_state(cfg, state: DMSState, events: set):
    """
    Turns frame-level events into stable alerts using consecutive-frame counters.
    Returns stable_alerts set: {"DROWSINESS","YAWN","DISTRACTION","PHONE"}
    """
    stable_alerts = set()

    # Drowsiness from consecutive closed-eye frames
    if "EYES_CLOSED" in events:
        state.closed_eye_counter += 1
    else:
        state.closed_eye_counter = max(0, state.closed_eye_counter - 2)

    if state.closed_eye_counter >= cfg.eye_closed_frames:
        stable_alerts.add("DROWSINESS")

    # Yawn
    if "YAWN" in events:
        state.yawn_counter += 1
    else:
        state.yawn_counter = max(0, state.yawn_counter - 2)

    if state.yawn_counter >= cfg.yawn_frames:
        stable_alerts.add("YAWN")

    # Distraction
    if "DISTRACTION" in events:
        state.distraction_counter += 1
    else:
        state.distraction_counter = max(0, state.distraction_counter - 2)

    if state.distraction_counter >= cfg.distraction_frames:
        stable_alerts.add("DISTRACTION")

    # Phone
    if "PHONE" in events:
        state.phone_counter += 1
    else:
        state.phone_counter = max(0, state.phone_counter - 2)

    if state.phone_counter >= cfg.phone_frames:
        stable_alerts.add("PHONE")

    state.active_alerts = stable_alerts
    return stable_alerts