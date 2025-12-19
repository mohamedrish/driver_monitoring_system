from ultralytics import YOLO

class YoloModule:
    """
    Pretrained model used here:
    3) YOLOv8 pretrained on COCO
    """
    def __init__(self, model_name="yolov8n.pt", conf=0.35):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, bgr_frame):
        # returns Ultralytics Results list
        results = self.model.predict(bgr_frame, conf=self.conf, verbose=False)
        return results[0]