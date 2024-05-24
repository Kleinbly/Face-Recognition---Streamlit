import cv2
import os
import uuid
import numpy as np


def LoadHaarCascades():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def ExtractFaces(file_path , scale_factor_value=1.2, min_neighbors_value=10, mode='per_image_file'):
    # Load the face cascade classifier
    face_cascade = LoadHaarCascades()

    img = cv2.imread(file_path)
    img = np.asarray(img)

    # Detect the faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(img, scaleFactor=float(scale_factor_value), minNeighbors=min_neighbors_value)

    detected_faces_array = []
    for (x, y, w, h) in faces:
        detected_face = img[y:y + h, x:x + w]
        try:
            detected_face = cv2.resize(detected_face, (160,160))
        except cv2.error as e:
            raise ValueError(f"Error resizing image: {e}")

        if mode == 'per_image_file':
            img_name = os.path.join(('temp_images/' + 'temp{}.jpg'.format(uuid.uuid1())))
            cv2.imwrite(img_name, detected_face)
            return img_name
        else:
            detected_faces_array.append(detected_face)

    if mode == 'real_time' and not len(detected_faces_array) == 0:
        return detected_faces_array

    return None

def FaceCheck(image):
    face_cascade = LoadHaarCascades()

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=9)

    number_detected_faces = 0
    for (x, y, h, w) in faces:
        number_detected_faces += 1

    return number_detected_faces





