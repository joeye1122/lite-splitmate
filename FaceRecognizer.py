import os
import json
import face_recognition
import numpy as np
import subprocess

import requests
from PIL import Image
from io import BytesIO

class FaceRecognizer:
    def __init__(self, json_file_path, photos_folder_path):

        self.json_file_path = json_file_path
        self.photos_folder_path = photos_folder_path
        self.known_encodings = []
        self.tenant_ids = []
        self.tenant_names = []

        self.utilities = []
        self.load_known_users()

    def get_face_encoding(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if len(face_encodings) == 0:
            print("No face detected.")
            return None
        return face_encodings[0]

    def load_known_users(self):

        # load json from local hardcoding file(for testing)
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        except Exception as e:
            print(f"unable to load cached")
            return

        users = data.get('users', [])

        for user in users:
            user_id = user['_id']
            user_name = user['name']
            photo_filename = f"{user_id}.jpg"
            photo_path = os.path.join(self.photos_folder_path, photo_filename)
            
            if os.path.exists(photo_path):
                image = face_recognition.load_image_file(photo_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    self.known_encodings.append(face_encodings[0])
                    self.tenant_ids.append(user_id)
                    self.tenant_names.append(user_name)
                else:
                    print(f"no face detacted on {user_name}")
            else:
                print(f"{user_name} didnt find {photo_path}")

        utilities = data.get('utilities', [])
        for utility in utilities:
            utility_id = utility['_id']
            utility_name = utility['name']
            mac_address = utility['mac_address']
            self.utilities.append({
                'id': utility_id,
                'name': utility_name,
                'mac_address': mac_address
            })


    def recognize_face(self, input_encoding, known_face_labels, known_face_encodings, threshold=0.6):
        distances = face_recognition.face_distance(known_face_encodings, input_encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < threshold:
            best_match_label = known_face_labels[best_match_index]
            print(f"Recognition result: {best_match_label}")
            return best_match_label
        else:
            print("Face not recognized.")
            return None
        
    def identify_tenant(self, input_image):

        input_encoding = self.get_face_encoding(input_image)
        return self.recognize_face(input_encoding, self.tenant_names, self.known_encodings, 0.6)


    def get_device_power(self, label):

        device_info = None
        for utility in self.utilities:
            if utility['name'].lower() == label.lower():
                device_info = utility
                break

        if not device_info:
            print(f"No utility found with label: {label}")
            return None

        mac_address = device_info['mac_address']

        script_path = 'ble_sensors.py'
        characteristic = 'power'
        command = 'read'

        try:
            result = subprocess.check_output([
                'python', script_path,
                '--mac_address', mac_address,
                '--command', command,
                '--characteristic', characteristic
            ], stderr=subprocess.STDOUT, text=True)

            result = result.strip()
            print(f"Device {label} ({mac_address}) power reading: {result}")
            return result

        except subprocess.CalledProcessError as e:
            print(f"Error calling script: {e.output}")
            return None