import cv2

def find_available_cameras(max_index=5):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

available_indexes = find_available_cameras()
print("avaliable cam index", available_indexes)
