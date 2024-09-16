from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
# model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)


# Load a pretrained YOLOv8n model
model = YOLO('yolov8m-pose.pt')

# Run inference on the source
results = model(source="WIN_20240830_14_23_50_Pro.mp4", show=True, conf=0.3, save=True)
