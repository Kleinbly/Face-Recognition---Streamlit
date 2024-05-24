import PIL
import numpy as np
import streamlit as st
import cv2
import os
import uuid
from utils import build_dataset
from Prediction import Load_Prediction_Object, predict_face
from utils_face import LoadHaarCascades
from utils import get_dataset

# Path: code\app.py
#
face_cascade = LoadHaarCascades()
#
model, encoder, embedder = Load_Prediction_Object()

database = get_dataset()

st.set_page_config(layout="wide")

st.sidebar.title("Settings")

# Create a menu bar
menu = ["Picture", "Webcam"]
choice = st.sidebar.selectbox("Input type", menu)
# Put slide to adjust tolerance
THRESHOLD = st.sidebar.slider("Threshold", 0.0, 1.0, 0.49, 0.01)
st.sidebar.info(
    "Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

# Infomation section
st.sidebar.title("Student Information")
name_container = st.sidebar.empty()
id_container = st.sidebar.empty()
name_container.info('Name: Unknown')
id_container.success('ID: Unknown')
if choice == "Picture":
    st.title("Face Recognition App")
    st.write('This app recognizes faces in a live video stream. To use it, simply press start and allow access to your webcam.')
    uploaded_images = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if len(uploaded_images) != 0:
        # Read uploaded image with face_recognition
        for image in uploaded_images:
            image_vs = PIL.Image.open(image)
            image_vs = np.array(image_vs)
            img_name = os.path.join(('temp_images/' + 'temp_loaded{}.jpg'.format(uuid.uuid1())))
            cv2.imwrite(img_name, image_vs)
            print(isinstance(image_vs, np.ndarray))

            # image, name, id = predict_face(model,encoder,embedder,image)
            index = predict_face(model,encoder,embedder,img_name)
            name = database[index]['name']
            os.remove(img_name)
            name_container.info(f"Name: {name}")
            id_container.success(f"ID: {index}")
            st.image(image_vs)
    else:
        st.info("Please upload an image")

elif choice == "Webcam":
    st.title("Face Recognition App")
    st.write('WEBCAM_PROMPT')
    # Camera Settings
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off the other app that is using the camera and restart app")
            st.stop()
        #image, name, id = recognize(frame, TOLERANCE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=7)
        for (x, y, w, h) in faces:
            detected_face = frame[y:y + h, x:x + w]
            score, index = predict_face(model, encoder, embedder, detected_face)
            name = 'Unknown'
            if index in database:
                name = database[index]['name']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.putText(frame, str(name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

            # Display name and ID of the person
            name_container.info(f"Name: {name}")
            id_container.success(f"ID: {str(round(score,4))}")
        FRAME_WINDOW.image(frame)

with st.sidebar.form(key='my_form'):
    st.title("Developer Section")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")