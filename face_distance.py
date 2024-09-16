from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
import cv2

# Load the model
model = YOLO("yolov8n.pt")
names = model.model.names

# Initialize the video capture
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening video file"

# Initialize distance calculation object
dist_obj = distance_calculation(names=names, view_img=True)

# Process video frames
while True:
    success, frame = cap.read()
    if not success:
        break

    # Run object tracking
    tracks = model.track(frame, persist=True)

    # Calculate distances and update the frame
    frame = dist_obj.start_process(frame, tracks)

    # Display the frame
    cv2.imshow('Distance Calculation', frame)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()