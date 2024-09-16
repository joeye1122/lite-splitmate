import torch
import numpy as np
import cv2
import json
from time import time
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self, capture_index):
        self.electronic_devices = ["laptop", "tv", "cell phone", "tablet", "refrigerator"]  # List of electronic devices

        self.capture_index = capture_index
        self.trigger_distance = 100  # 设定触发距离阈值，单位为像素

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model
        model.fuse()
        return model

    def predict(self, frame):
        # Enable tracker
        results = self.model.track(frame, tracker="bytetrack.yaml")
        return results

    def trigger_device_use(self, person, device, distance):
        """ 当人靠近设备时，触发这个函数生成 JSON """
        data = {
            "person_id": person,  # Placeholder for the person identifier
            "device_id": device,  # Placeholder for the device identifier
            "distance": distance,
            "status": "device in use",  # Placeholder for usage status
            "timestamp": time()  # 当前的时间戳
        }
        
        # 生成 JSON 并输出（在实际应用中可能会保存到文件或发送到服务器）
        json_data = json.dumps(data, indent=4)
        print(json_data)

    def plot_bboxes(self, results, frame):
        persons = []  # To store the person center coordinates
        devices = []  # To store the electronic devices center coordinates

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            confidences = boxes.conf
            class_ids = boxes.cls.astype(int)

            # Get track id
            track_ids = boxes.id  # Get the track IDs if available

            # For every target in the result
            for i, (xyxy, confidence, class_id) in enumerate(zip(xyxys, confidences, class_ids)):
                # 提取边界框的坐标
                x1, y1, x2, y2 = map(int, xyxy)
                # 获取类别名称
                label = self.CLASS_NAMES_DICT[class_id]

                # Track id
                if track_ids is not None:
                    track_id = int(track_ids[i])
                    label_text = f'{label} {confidence:.2f} ID: {track_id}'  # Include track ID in label
                else:
                    label_text = f'{label} {confidence:.2f}'

                # Center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 在边界框内绘制对角线
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x2, y1), (0, 255, 255), 2)

                # If label as person, save the coord of center
                if label == "person":
                    persons.append((center_x, center_y, track_id))  # Add track_id to persons list
                    print(f"Person center: {center_x}, {center_y}")

                # If label as electronic device, save the coord of center
                if label in self.electronic_devices:
                    devices.append((center_x, center_y, label))  # Add device label to devices list
                    print(f"{label} center: {center_x}, {center_y}")

        # Calculating distance between person and electronic devices and trigger function
        for person_center in persons:
            for device_center in devices:
                # Calculation distance between person and device (from center)
                distance = np.sqrt((person_center[0] - device_center[0]) ** 2 +
                                   (person_center[1] - device_center[1]) ** 2)
                print(f"Distance from person to device: {distance:.2f} pixels")

                # If distance is less than the trigger distance, trigger device use function
                if distance < self.trigger_distance:
                    self.trigger_device_use(person_center[2], device_center[2], distance)  # Trigger function

                # Plot person center to electronic center
                cv2.line(frame, (person_center[0], person_center[1]), (device_center[0], device_center[1]), (255, 0, 0), 2)  # 蓝色线

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)

        # cap = cv2.VideoCapture("Movie on 30-8-2024 at 12.54 PM.mov")

        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)

            # Bounding box
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            # Show FPS
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = ObjectDetection(capture_index=0)
    detector()


if __name__ == "__main__":
    main()
