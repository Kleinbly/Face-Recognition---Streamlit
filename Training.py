from utils_training_and_prediction import load_classes
from utils_training_and_prediction import getBatches
from utils_training_and_prediction import get_embedding
from utils import build_dataset

import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def training_model():
    print('STEP 1 : Initializing X and Y arrays.')
    X = []
    Y = []
    Z = []

    print('STEP 2 : Load Images & Classes.')
    X, Y, Z, I = load_classes('Dataset/Registered_Images')

    build_dataset(X,Y,Z,I)

    print('STEP 3 : Image augmentation & Batch generation')
    X_batch, Y_batch = getBatches(X,Z)

    print('STEP 4 : Configure FACENET')
    embedder = FaceNet()

    # Derive embeddings
    EMBEDDED_X = []

    for img in X_batch:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        EMBEDDED_X.append(get_embedding(embedder, img))
    EMBEDDED_X = np.asarray(EMBEDDED_X)


    print('STEP 5 : Encode Labels')
    encoder = LabelEncoder()
    encoder.fit(Y_batch)
    Y_batch = encoder.transform(Y_batch)


    print('STEP 6 : Split the dataset')
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_batch, test_size=0.3, random_state=17)

    print('STEP 7 : Initialize SVM model')
    model = SVC(probability=True)

    print('STEP 8 : Set the grid search')
    # Set grid search
    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear']
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters found: ", best_params)
    print("Best model found: ", best_model)

    print('STEP 9 : Evaluate the best model on the test set')
    ypreds_train = best_model.predict(X_train)
    ypreds_test = best_model.predict(X_test)

    # Measure accuracy
    print('Accuracy (Training):', accuracy_score(Y_train, ypreds_train))
    print('Accuracy (Test):', accuracy_score(Y_test, ypreds_test))

    print('STEP 10 : Save the model & encoder')
    # Save the trained model to a file using pickle
    with open('SerializedObjects/svc_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    with open('SerializedObjects/encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    print('\n END.')
