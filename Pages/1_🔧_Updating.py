import streamlit as st
import cv2
import os
import uuid
import numpy as np
from PIL import Image
from utils import submitNew
from utils import get_info_from_id
from utils import deleteUser
from utils_face import FaceCheck

from Training import training_model

st.set_page_config(layout="wide")
st.title("Face Recognition App")
st.write("This app is used to add new faces to the dataset")

menu = ["Adding", "Deleting", "Adjusting"]
choice = st.sidebar.selectbox("Options", menu)


if choice == "Adding":
    name = st.text_input("Name", placeholder='Enter name')
    # Create 2 options: Upload image or use webcam
    # If upload image is selected, show a file uploader
    # If use webcam is selected, show a button to start webcam
    upload = st.radio("Upload image or use webcam", ("Upload", "Webcam"))

    if upload == "Upload":
        uploaded_image = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
        if uploaded_image is not None:
            st.image(uploaded_image)
            submit_btn = st.button("Submit", key="submit_btn")
            if submit_btn:
                if name == "":
                    st.error("Please enter name")
                else:

                    image = Image.open(uploaded_image)
                    # Convert the image to a numpy array
                    image_array = np.array(image)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                    ret = submitNew(name=name, id=id, image=image_array)
                    if ret == 1:
                        # Train the model #TODO
                        with st.spinner(text="In progress..."):
                            training_model()
                        st.success("Person Added")
                    elif ret == 0:
                        st.error("Multiple faces detected")
                    elif ret == -1:
                        st.error("There is no face in the picture")

    elif upload == "Webcam":
        img_file_buffer = st.camera_input("Take a picture")
        submit_btn = st.button("Submit", key="submit_btn")
        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if submit_btn:
                if name == "":
                    st.error("Please enter name")
                else:
                    # Convert image to np.array
                    uploaded_image = np.asarray(cv2_img)
                    ret = submitNew(name, id, uploaded_image)
                    
                    if ret == 1:
                        # Train the model #TODO
                        with st.spinner(text="In progress..."):
                            training_model()
                        st.success("Person Added")
                    elif ret == 0:
                        st.error("Multiple faces detected")
                    elif ret == -1:
                        st.error("There is no face in the picture")

elif choice == "Deleting":
    def del_btn_callback(image_file_path):
        deleteUser(image_file_path)
        # Train the model #TODO
        with st.spinner(text="In progress..."):
            training_model()
        st.success("User deleted")


    id = st.text_input("ID", placeholder='Enter id')

    submit_btn = st.button("Submit", key="submit_btn")
    if submit_btn:
        name, image_path = get_info_from_id(id)
        if name == None and image_path == None:
            st.error("User ID does not exist")
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.success(f"Name of student with ID {id} is: {name}")
            st.warning("Please check the image below to make sure you are deleting the right student")
            st.image(img)
            del_btn = st.button("Delete", key="del_btn", on_click=del_btn_callback, args=(image_path,))
  
elif choice == "Adjusting":
    def form_callback(id, old_name, old_image, new_name, new_image):
        new_name = st.session_state['new_name']
        new_image = st.session_state['new_image']

        name = old_name
        image = old_image

        if new_image is not None:
            image = cv2.imdecode(np.frombuffer(new_image.read(), np.uint8), cv2.IMREAD_COLOR)

        if new_name != old_name:
            name = new_name

            old_img = cv2.imread(old_image_path)
            old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)

            # TODO Set a way for old and new images cases

            file_path_parts = old_image.split('/')
            old_path = ''.join(file_path_parts[:2])
            print(old_path)
            new_path = ''.join(file_path_parts[:1])
            new_path = new_path.join(name)
            print(new_path)

            try:
                os.rename(old_path, new_path)
            except OSError as e:
                print(f"Error: {e}")
            

        ret = FaceCheck(image)
        if ret == 1:
            cv2.imwrite(old_image, image)
            # Train the model #TODO
            with st.spinner(text="In progress..."):
                training_model()
            st.success("User Added")
        elif ret == 0:
            st.error("Multiple faces detected")
        elif ret == -1:
            st.error("There is no face in the picture")



    id = st.text_input("ID", placeholder='Enter id')
    submit_btn = st.button("Submit", key="submit_btn")
    if submit_btn:
        old_name, old_image_path = get_info_from_id(id)
        if old_name == None and old_image_path == None:
            st.error("Student ID does not exist")
        else:
            old_img = cv2.imread(old_image_path)
            old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)

            with st.form(key='my_form'):
                st.title("Adjusting user info")
                col1, col2 = st.columns(2)
                new_name = col1.text_input("Name", key='new_name', value=old_name, placeholder='Enter new name')
                new_image = col1.file_uploader("Upload new image", key='new_image', type=['jpg', 'png', 'jpeg'])
                col2.image(old_img, caption='Current image', width=400)
                st.form_submit_button(label='Submit', on_click=form_callback, args=(id, old_name, old_image_path, new_name, new_image))

