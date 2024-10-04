import torch
import numpy as np
import cv2
import json
from time import strftime, localtime, time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import face_recognition

import base64
from io import BytesIO
from PIL import Image



tenants = []
utilities = []

def fetch_data():
    global tenants, utilities

    with open('example.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        
    # url = 'http://localhost:5000/read'
    # response = requests.get(url)
    # response.raise_for_status()
    # data = response.json()

    tenants = data.get('tenants', [])
    utilities = data.get('utilities', [])


def identify_tenant(input_image_path):
    global tenants
    known_encodings = []
    tenant_ids = []
    tenant_names = []

    # 1. for every tenant
    for tenant in tenants:
        tenant_id = tenant['id']
        tenant_name = tenant['name']
        tenant_photo_base64 = tenant['photo']  # Base64 encoding JPEG

        try:
            # decode Base64 string then load image
            tenant_image_data = base64.b64decode(tenant_photo_base64)
            tenant_image = Image.open(BytesIO(tenant_image_data))
            tenant_image_np = np.array(tenant_image)

            tenant_face_encodings = face_recognition.face_encodings(tenant_image_np)
            if len(tenant_face_encodings) > 0:
                tenant_face_encoding = tenant_face_encodings[0]
                known_encodings.append(tenant_face_encoding)
                tenant_ids.append(tenant_id)
                tenant_names.append(tenant_name)
            else:
                print(f"未在租户 {tenant_name} 的照片中检测到人脸。")
        except Exception as e:
            print(f"处理租户 {tenant_name} 的照片时发生错误：{e}")

    # 检查是否成功加载了租户的特征向量
    if not known_encodings:
        print("没有可用于比较的租户人脸特征向量。")
        return None

    # 2. 加载输入图像并提取特征向量
    try:
        input_image = face_recognition.load_image_file(input_image_path)
        input_face_locations = face_recognition.face_locations(input_image)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    except Exception as e:
        print(f"加载输入图像时发生错误：{e}")
        return None

    if len(input_face_encodings) == 0:
        print("未能在输入图像中检测到人脸。")
        return None

    input_face_encoding = input_face_encodings[0]  # 假设只处理第一张人脸

    # 3. 比较输入人脸与租户人脸特征向量
    distances = face_recognition.face_distance(known_encodings, input_face_encoding)
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    threshold = 0.6  # 可根据需要调整

    if min_distance < threshold:
        matched_tenant = tenants[min_distance_index]
        matched_tenant_name = tenant_names[min_distance_index]
        print(f"识别结果：{matched_tenant_name}，距离：{min_distance}")
        return matched_tenant
    else:
        print("无法识别此人脸。")
        return None
    


class ObjectDetection:

    def __init__(self, capture_index):
        self.electronic_devices = ["laptop", "tv", "cell phone", "tablet", "refrigerator"]  # List of electronic devices

        self.capture_index = capture_index
        self.enter_distance = 100  # the threshold for entering the device distance
        self.exit_distance = 130  # the threshold for exiting the device distance
        self.trigger_duration = 2  # time threshold in sec
        self.usage_status = {}  # record the person_id and device_id status
        self.last_update_time = {}  # record the person and device's last update timestampe

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model
        model.fuse()
        return model
    
    def predict(self, frame):
        # Enable tracker with bytetracker
        results = self.model.track(frame, tracker="bytetrack.yaml")
        return results

    def record_device_usage(self, person_id, device_label, start_time, end_time):
        """ 
        the json to be generated when the event is finished
        """
        data = {
            "person_id": person_id,  # Placeholder for the person identifier
            "device_id": device_label,  # Placeholder for the device identifier
            "start_time": start_time,
            "end_time": end_time
        }

        json_data = json.dumps(data, indent=4)
        print(json_data)

    def plot_bboxes(self, results, frame):
        persons = []  # To store the person center coordinates
        devices = []  # To store the electronic devices center coordinates

        persons_position = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            confidences = boxes.conf
            class_ids = boxes.cls.astype(int)

            # Get track id
            track_ids = boxes.id  # Get the track IDs if available

            # For every target in the result
            for i, (xyxy, confidence, class_id) in enumerate(zip(xyxys, confidences, class_ids)):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                # Get class label
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

                # If label is "person", save center coordinates and check if using a device
                if label == "person":
                    persons.append((center_x, center_y, track_id))  # Add track_id to persons list
                    print(f"Person center: {center_x}, {center_y}")
                    persons_position.append((x1, y1, x2, y2, frame,track_id))

                # If label is an electronic device, save center coordinates
                if label in self.electronic_devices:
                    devices.append((center_x, center_y, label))  # Add device label to devices list
                    print(f"{label} center: {center_x}, {center_y}")

        # Calculate distance between person and devices, then apply logic for usage
        for person_center in persons:
            for device_center in devices:
                # Calculate distance from person to device (center to center)
                distance = np.sqrt((person_center[0] - device_center[0]) ** 2 +
                                (person_center[1] - device_center[1]) ** 2)
                print(f"Distance from person to device: {distance:.2f} pixels")

                person_id = person_center[2]
                device_label = device_center[2]
                current_time = time()

                # Enter logic: distance is less than enter_distance
                if distance < self.enter_distance:
                    # Check if it's the first time entering the trigger distance (no start_time in usage_status)
                    if (person_id, device_label) not in self.usage_status:
                        # Start the timer and record the starting time
                        self.usage_status[(person_id, device_label)] = {"starting_time": current_time}
                        print(f"Starting timer for {person_id} and {device_label}")
                    else:
                        # If the timer is already started, check if duration is greater than trigger_duration
                        duration = current_time - self.usage_status[(person_id, device_label)]["starting_time"]

                        if duration >= self.trigger_duration and "in_use" not in self.usage_status[(person_id, device_label)]:

                            for person_l in persons_position:
                                crop_origi_mage(person_l[0],person_l[1],person_l[2],person_l[3],person_l[4],person_l[5])


                            # If the duration exceeds trigger_duration, mark the device as in use
                            self.usage_status[(person_id, device_label)]["in_use"] = True
                            start_time = strftime("%Y-%m-%d %H:%M:%S", localtime(self.usage_status[(person_id, device_label)]["starting_time"]))
                            self.usage_status[(person_id, device_label)]["start_time"] = start_time
                            print(f"Device {device_label} is now in use by {person_id}")


                # Exit logic: distance is greater than exit_distance
                elif distance > self.exit_distance:
                    # If the person was using the device, end the usage session
                    if (person_id, device_label) in self.usage_status and "in_use" in self.usage_status[(person_id, device_label)]:
                        end_time = strftime("%Y-%m-%d %H:%M:%S", localtime(time()))
                        start_time = self.usage_status[(person_id, device_label)]["start_time"]
                        # Generate JSON and remove usage status
                        self.record_device_usage(person_id, device_label, start_time, end_time)
                        del self.usage_status[(person_id, device_label)]
                        print(f"Usage ended for {person_id} on {device_label}")





                # Plot line from person center to device center
                cv2.line(frame, (person_center[0], person_center[1]), (device_center[0], device_center[1]), (255, 0, 0), 2)

        return frame



    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)

        # cap = cv2.VideoCapture("IMG_3534.MOV")

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

def crop_origi_mage(x1, y1, x2, y2, orig_img, track_id):
    """ 
    Crops the original image based on the coordinates and saves it in a folder named by track_id.
    The image is saved with the current timestamp as its filename.
    """
    # Crop the image based on the coordinates
    cropped_img = orig_img[y1:y2, x1:x2]
    
    # Create a folder named after the track ID if it doesn't exist
    folder_name = f"track_{track_id}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Generate the file name with current timestamp
    current_time = strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_name, f"{current_time}.png")
    
    # Save the cropped image
    cv2.imwrite(file_path, cropped_img)
    print(f"Cropped image saved at: {file_path}")


def main():
    detector = ObjectDetection(capture_index=0)
    
    detector()


if __name__ == "__main__":
    main()

