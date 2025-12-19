import cv2
import time

from config import DMSConfig
from face_module import FaceModule
from yolo_module import YoloModule
from alerts import AlertManager
from dms_logic import DMSState, compute_events, update_state
from utils import estimate_head_pose

def draw_overlay(frame, face_box, metrics, alerts, cfg):
    h, w = frame.shape[:2]

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 80), 2)

    # YOLO phone boxes
    for (x1, y1, x2, y2) in metrics.get("phone_boxes", []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 220), 2)
        cv2.putText(frame, "cell phone", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 220), 2)

    # Alerts banner
    if alerts:
        banner = "ALERT: " + ", ".join(sorted(list(alerts)))
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 255), -1)
        cv2.putText(frame, banner, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if cfg.show_debug:
        y = 80
        ear = metrics.get("EAR", None)
        mar = metrics.get("MAR", None)
        yaw = metrics.get("yaw", None)
        pitch = metrics.get("pitch", None)

        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f} (thr {cfg.ear_threshold})", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            y += 30
        if mar is not None:
            cv2.putText(frame, f"MAR: {mar:.3f} (thr {cfg.mar_threshold})", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            y += 30
        if yaw is not None and pitch is not None:
            cv2.putText(frame, f"HeadPose pitch={pitch:.1f}, yaw={yaw:.1f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            y += 30

def main():
    cfg = DMSConfig()

    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    face = FaceModule(det_conf=cfg.face_det_conf, track_conf=cfg.face_track_conf)
    yolo = YoloModule(model_name=cfg.yolo_model, conf=cfg.yolo_conf)
    alerter = AlertManager(min_interval_sec=1.0)
    state = DMSState()

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Face box (fast) + landmarks (detailed)
        face_box = face.detect_face(frame)
        landmarks = face.get_landmarks(frame)

        metrics = {"phone_boxes": []}
        stable_alerts = set()

        if landmarks is not None:
            h, w = frame.shape[:2]
            head_pose = estimate_head_pose(landmarks, w, h)
            
            if head_pose is not None and not state.calibrated:
               pitch, yaw, roll = head_pose
               state.calib_sum_yaw += yaw
               state.calib_sum_pitch += pitch
               state.calib_samples += 1

               # calibrate after ~40 frames (~2 seconds at 20 FPS)
               if state.calib_samples >= 40:
                   state.pose_bias_yaw = state.calib_sum_yaw / state.calib_samples
                   state.pose_bias_pitch = state.calib_sum_pitch / state.calib_samples
                   state.calibrated = True

            # YOLO detections (for phone)
            yolo_result = yolo.detect(frame)
            metrics, frame_events = compute_events(cfg, state, landmarks, head_pose, yolo_result, face_box)
        
            metrics, frame_events = compute_events(cfg, state, landmarks, head_pose, yolo_result, face_box)
            stable_alerts = update_state(cfg, state, frame_events)

            # Trigger sounds
            for a in stable_alerts:
                alerter.alert(a)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        draw_overlay(frame, face_box, metrics, stable_alerts, cfg)
        cv2.imshow("Driver Monitoring System (3 Pretrained Models)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()