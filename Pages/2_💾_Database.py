import cv2
import streamlit as st
import numpy as np

from utils import get_dataset


st.title('Registered Faces')
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')

#Load database
database = get_dataset()

# Name the columns
Index, Id, Name, Image = st.columns(spec=[1, 1, 2, 2], gap='large')
with Index:
    st.subheader('Number')
with Id:
    st.subheader('ID Number')
with Name:
    st.subheader('Names')
with Image:
    st.subheader('Images')

# Display the registered faces (Images + Identity)
for idx, person in database.items():
    st.write('------------------------------------------------------------------------------------')
    Index, Id, Name, Image = st.columns(spec=[1, 3, 3, 3], gap='medium')
    with Index:
        st.write(person['counter'])
    with Id:
        st.write(idx)
    with Name:
        st.write(person['name'])
    with Image:
        img = cv2.imread(person['image_filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        st.image(img, width=165)
