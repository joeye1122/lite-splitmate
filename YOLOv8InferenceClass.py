import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO


class ObjectDetection:

    def __init__(self, capture_index):
        self.electronic_devices = ["laptop", "tv", "cell phone", "tablet", "refrigerator"]  # List of electronic devices

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model
        model.fuse()
        return model

    def predict(self, frame):
        #enable tracker
        # results = self.model(frame, tracker=True)
        results = self.model.track(frame, tracker="bytetrack.yaml")
        return results

    def plot_bboxes(self, results, frame):
        persons = []  # To store the person center coordinates
        devices = []  # To store the electronic devices center coordinates

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            confidences = boxes.conf
            class_ids = boxes.cls.astype(int)


            #get track id
            track_ids = boxes.id  # Get the track IDs if available


            # for every target in the result
            for i, (xyxy, confidence, class_id) in enumerate(zip(xyxys, confidences, class_ids)):
                # 提取边界框的坐标
                x1, y1, x2, y2 = map(int, xyxy)
                # 获取类别名称
                label = self.CLASS_NAMES_DICT[class_id]

                # track id
                if track_ids is not None:
                    track_id = int(track_ids[i])
                    label_text = f'{label} {confidence:.2f} ID: {track_id}'  # Include track ID in label
                else:
                    label_text = f'{label} {confidence:.2f}'

                # center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 在边界框内绘制对角线
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x2, y1), (0, 255, 255), 2)

                # if label as person, save the coord of center
                if label == "person":
                    persons.append((center_x, center_y))
                    print(f"Person center: {center_x}, {center_y}")

                # if label as electronic device, save the coord of center
                if label in self.electronic_devices:
                    devices.append((center_x, center_y))
                    print(f"{label} center: {center_x}, {center_y}")

        # calculating distance between person and electronic devices and plot the line
        for person_center in persons:
            for device_center in devices:
                # calculation disatance between person and device(from center)
                distance = np.sqrt((person_center[0] - device_center[0]) ** 2 +
                                (person_center[1] - device_center[1]) ** 2)
                print(f"Distance from person to device: {distance:.2f} pixels")

                # polt person center to electronic center
                cv2.line(frame, person_center, device_center, (255, 0, 0), 2)  # 蓝色线

        return frame


    def __call__(self):
        # cap = cv2.VideoCapture(self.capture_index)

        cap = cv2.VideoCapture("IMG_3534.MOV")

        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)

            # bounding box
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            # show FPS
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
