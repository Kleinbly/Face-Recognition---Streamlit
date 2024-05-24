import os
import cv2
import uuid
import numpy as np

from utils_face import ExtractFaces
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load images and extract faces
def load_faces(folder_path):
    # Initialize an empty list to store face images
    faces = {}

    # Iterate over all files in the given folder
    for file in os.listdir(folder_path):
        # Check if the current item is a file
        if os.path.isfile(folder_path + '/' + file):
            # Construct the full path to the file
            path = folder_path + '/' + file

            # Extract faces from the image at the given path
            face_img_path = ExtractFaces(path)

            # If face extraction was successful (i.e., a face was found), add it to the faces list
            if face_img_path is not None:
                faces[face_img_path] = path

    # Return the list of paths (face images)
    return faces

def load_classes(folder_path):
    # Initialize empty lists to store face images (X) and corresponding labels (Y)
    # X and Y store objects of type numpy.array
    X = []
    Y = []
    Z = []
    I = []

    # Iterate over each sub-directory in the given folder
    for sub_dir in os.listdir(folder_path):
        # Construct the full path to the sub-directory
        path = folder_path + '/' + sub_dir + '/'
        print(sub_dir, 'Actual folder')
        # Load faces from the sub-directory
        face_dict = load_faces(path)

        detected_faces = face_dict.keys()

        # Create labels for the detected faces
        # The label is the name of the sub-directory, repeated for each detected face
        labels = [sub_dir for _ in range(len(detected_faces))]

        # Generate uuids indexes for each detected face
        indexes = [uuid.uuid1() for _ in range(len(detected_faces))]

        # Load image paths for later
        image_Array = face_dict.values()

        # Extend the face images list (X) with the detected faces
        X.extend(detected_faces)

        # Extend the labels list (Y) with the created labels
        Y.extend(labels)

        # Extend the uuid index list (Z) with created indexes
        Z.extend(indexes)

        # d
        I.extend(image_Array)

    # Convert the lists to numpy arrays for efficient numerical operations
    return np.asarray(X), np.asarray(Y), np.asarray(Z), np.asarray(I)

# Image augmentation
def getBatches(X,Y, number_of_batches=45):
    # Create an instance of ImageDataGenerator with suitable augmentations
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,  # Rotate the image up to 12.5 degrees
        width_shift_range=0.2,  # Shift the image horizontally by up to 20%
        height_shift_range=0.,  # Shift the image vertically by up to 20%
        brightness_range=(0.8, 1.3),
        shear_range=0.15,  # Shear the image by 10%
        horizontal_flip=True,  # Allow horizontal flipping
        fill_mode='nearest'  # Strategy for filling newly created pixels
    )

    # Assuming you have a dataset of face images loaded as a NumPy array `faces`
    # and corresponding labels `labels`
    # Here's how to create an iterator that generates batches of augmented images

    # Suppose `faces` is of shape (n_samples, height, width, channels)
    # and `labels` is a list or array of shape (n_samples,)
    # You would typically have this data split into training and validation sets

    X_IMAGES = []
    for img_path in X:
        img = cv2.imread(img_path)
        # Delete the file
        os.remove(img_path)

        X_IMAGES.append(img)
    X_IMAGES = np.asarray(X_IMAGES)

    augmented_data = data_gen.flow(X_IMAGES, Y, batch_size=30, shuffle=True)  # Adjust batch_size as per your requirement


    # Get one batch of data from DATA AUGMENTATION
    X_batch , Y_batch = next(augmented_data)
    for _ in range(number_of_batches):
        X_TEMP , Y_TEMP = next(augmented_data)
        X_batch = np.concatenate((X_batch, X_TEMP))
        Y_batch = np.concatenate((Y_batch, Y_TEMP))


    return X_batch, Y_batch

# GET EMBEDDINGS FROM FACENET
def get_embedding(embedder,face_img):
    if face_img is None:
        return None

    if not isinstance(face_img, np.ndarray):
        img = cv2.imread(face_img)

        # Delete the temp file
        os.remove(face_img)

        # OpenCV loads images in BGR format by default
        face_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        #img_name = os.path.join(('temp_images/' + 'LUV{}.jpg'.format(uuid.uuid1())))
        #cv2.imwrite(img_name, face_img)


    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img.astype('float32')


    yhat = embedder.embeddings(face_img)
    return yhat[0]











