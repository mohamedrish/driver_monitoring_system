from dataclasses import dataclass

@dataclass
class DMSConfig:
    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Face detection / mesh confidence
    face_det_conf: float = 0.5
    face_track_conf: float = 0.5

    # Drowsiness thresholds (tune per camera/lighting)
    ear_threshold: float = 0.23          # Eye Aspect Ratio threshold (lower => eyes closed)
    eye_closed_frames: int = 18          # consecutive frames to confirm drowsiness

    # Yawn thresholds
    mar_threshold: float = 0.60          # Mouth Aspect Ratio threshold (higher => mouth open)
    yawn_frames: int = 10

    # Distraction (head pose) thresholds
    yaw_threshold_deg: float = 35.0      # looking left/right
    pitch_threshold_deg: float = 25.0    # looking down/up (we'll mainly use absolute pitch)

    distraction_frames: int = 18         # consecutive frames to confirm distraction

    # YOLO object detection
    yolo_model: str = "yolov8n.pt"       # pretrained COCO
    yolo_conf: float = 0.35
    phone_class_name: str = "cell phone"
    phone_frames: int = 8

    # UI
    show_debug: bool = True