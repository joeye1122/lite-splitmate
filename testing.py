import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
results = model.predict(source="0", show=True)  # Display preds. Accepts all YOLO predict arguments

