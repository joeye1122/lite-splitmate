import face_recognition
import numpy as np
import os
import json

def get_face_encoding(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if len(face_encodings) == 0:
        print("No face detected.")
        return None
    return face_encodings[0]

def load_known_faces(json_file, images_folder):
    known_face_encodings = []
    known_face_labels = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            users = data.get('users', [])

        for user in users:
            user_id = user.get('_id')
            user_name = user.get('name', 'Unknown')
            image_filename = os.path.join(images_folder, f"{user_id}.jpg")

            if not os.path.isfile(image_filename):
                print(f"Image file not found for user {user_name}: {image_filename}")
                continue

            image = face_recognition.load_image_file(image_filename)
            face_encoding = get_face_encoding(image)
            if face_encoding is not None:
                known_face_encodings.append(face_encoding)
                known_face_labels.append(user_id)  # or use user_name
            else:
                print(f"No face detected in image for user {user_name}.")
    except Exception as e:
        print(f"Error loading known face data: {e}")

    return known_face_labels, known_face_encodings

def recognize_face(input_encoding, known_face_labels, known_face_encodings, threshold=0.6):
    distances = face_recognition.face_distance(known_face_encodings, input_encoding)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < threshold:
        best_match_label = known_face_labels[best_match_index]
        print(f"Recognition result: {best_match_label}")
        return best_match_label
    else:
        print("Face not recognized.")
        return None

def main():
    # Define paths for the JSON file and image folder
    json_file = 'cached_data.json'
    images_folder = 'user_images'

    # Load known face data
    known_face_labels, known_face_encodings = load_known_faces(json_file, images_folder)
    if not known_face_encodings:
        print("No known face data loaded.")
        return

    # Load the input image
    input_image_path = 'input.jpg'  # Replace with your input image path
    image = face_recognition.load_image_file(input_image_path)

    # Get the face encoding for the input image
    input_encoding = get_face_encoding(image)
    if input_encoding is None:
        return

    # Recognize the face
    recognize_face(input_encoding, known_face_labels, known_face_encodings)

if __name__ == '__main__':
    main()
