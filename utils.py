import os
import cv2
import uuid
import numpy as np
import pickle as pkl
from collections import defaultdict
from utils_face import FaceCheck



from utils_training_and_prediction import load_classes

information = defaultdict(dict)

def get_dataset():
    with open('SerializedObjects/database.pkl', 'rb') as f:
        return pkl.load(f)
    
def build_dataset(X,Y,Z,I):
    for index in range(len(X)):
        # image_name = X[index].split('/')[-1].split('.')[0]

        person_id = Z[index]
        person_image = cv2.imread(X[index])
        file_path = I[index]
        person_name = Y[index]

        add_to_dataset(database=information, id=person_id, image=person_image, path_to_image=file_path, name=person_name)

    with open(os.path.join('SerializedObjects/database.pkl'), 'wb') as file:
        pkl.dump(information, file)

def add_to_dataset(database, id, image, path_to_image, name):
    database[id] = {'image': image,
                    'image_filepath' : path_to_image,
                    'counter': len(database),
                    'name': name}

def submitNew(name, image, id):
    database = get_dataset()
    

    isFaceInPic = FaceCheck(image)
    if isFaceInPic == 1:
        # One face was detected
        # Add the image to the Registered_Images directory
        directory_path = os.path.join('Dataset/' + 'Registered_Images/' + name + '/')

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        filename = os.path.join(directory_path + f'new_{uuid.uuid1()}.jpg')
        cv2.imwrite(filename, image)

        return 1
    elif isFaceInPic == 0:
        # No face was detected
        return -1
    elif isFaceInPic > 1:
        # Multiple faces detected
        return 0
    
def get_info_from_id(id):
    database = get_dataset()

    
    keys = list(database.keys())
    # Convert UUID keys to string  
    str_keys = [str(key) for key in keys]

    print(id)

    if id in str_keys:
        person_array = list(database.values())
        print(person_array)
        person = person_array[str_keys.index(id)]

        name = person['name']
        image = person['image_filepath']
        return name, image
        
    return None, None


def deleteUser(file_path):
    os.remove(file_path)
