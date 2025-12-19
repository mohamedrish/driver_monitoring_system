# Driver Monitoring System (DMS) - 3 Pretrained Models

## Overview
This project monitors a driver's state in real time and issues alerts for:
- Drowsiness (prolonged eye closure)
- Yawning (mouth opening over time)
- Distraction (head pose looking away)
- Phone usage (cell phone detected by object detection)

## Pretrained Models Used
1. MediaPipe Face Detection (pretrained)
2. MediaPipe Face Mesh (pretrained)
3. YOLOv8n (COCO pretrained) via Ultralytics

## Install
pip install -r requirements.txt

## Run
python src/main.py

## Notes
- Thresholds may need tuning depending on camera position and lighting.
- The YOLO model detects "cell phone" from COCO classes.