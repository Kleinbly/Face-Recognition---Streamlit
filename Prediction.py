import pickle
import numpy as np

from keras_facenet import FaceNet
from utils_training_and_prediction import get_embedding
from utils_face import ExtractFaces


def Load_Prediction_Object():
    # Load the model and encoder from their respective files using pickle
    with open('SerializedObjects/svc_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('SerializedObjects/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    embedder = FaceNet()

    return model,encoder,embedder
def predict_face(model, encoder, embedder, img_file_path):
    T = []
    if not isinstance(img_file_path, np.ndarray):
        TEST_EMBEDDING = get_embedding(embedder,ExtractFaces(img_file_path))
    else:
        TEST_EMBEDDING = get_embedding(embedder, img_file_path)
    if TEST_EMBEDDING is not None:
        T.append(TEST_EMBEDDING)
        T_EMBEDDING = np.asarray(T)
        probabilities = model.predict_proba(T_EMBEDDING)
        prob_score = (np.max(probabilities, axis=1))
        pred_test = model.predict(T_EMBEDDING)
        if prob_score[0] < 0.49:
            return 0,'Unknown'
        else:
            return prob_score[0] ,encoder.inverse_transform(pred_test)[0]
    else:
        return 0,'No face was detected'


#model, encoder, embedder = Load_Prediction_Object()
#print('Billy:::')
#predict_face(model, encoder, embedder, 'Test/IMG_20221004_235312_695.jpg')
#print('Jenna:::')
#predict_face(model, encoder, embedder, 'Test/jenna-ortega-2.png')
#print('Jenna:::')
#predict_face(model, encoder, embedder, 'Test/jenna_valid.jpg')
#print('Billy:::')
#predict_face(model, encoder, embedder, 'Test/Photo.jpg')
#print('Elon:::')
#predict_face(model, encoder, embedder, 'Test/elon_valid.jpeg')
#print('Elon:::')
#predict_face(model, encoder, embedder ,'Test/elon-musk3.jpeg')
#print('Eric:::')
#predict_face(model, encoder, embedder ,'Test/visu_1515061157.jpg')
