import mediapipe as mp
import cv2

class FaceModule:
    """
    Pretrained models used here:
    1) MediaPipe Face Detection
    2) MediaPipe Face Mesh
    """
    def __init__(self, det_conf=0.5, track_conf=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=det_conf
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # better eyes/iris landmarks
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def detect_face(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self.face_detection.process(rgb)
        if not res.detections:
            return None

        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = bgr_frame.shape[:2]
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = x1 + bw
        y2 = y1 + bh
        # clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        return (x1, y1, x2, y2)

    def get_landmarks(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        face_landmarks = res.multi_face_landmarks[0]
        h, w = bgr_frame.shape[:2]
        pts = []
        for lm in face_landmarks.landmark:
            pts.append((int(lm.x * w), int(lm.y * h)))
        return pts  # list of 468+ points in pixel coords