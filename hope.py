import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model('digits.model.a')

# Tiền xử lý ảnh trước khi đưa vào mô hình
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image

# Tạo giao diện ứng dụng Streamlit
st.title('Nhận diện chữ số viết tay')

uploaded_image = st.file_uploader('Tải ảnh lên', type=['png','jpg','jpeg'])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Ảnh đã tải lên:')
    # Tiền xử lý ảnh và dự đoán kết quả
    image = Image.open(uploaded_image)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    # Hiển thị kết quả dự đoán
    digit = np.argmax(prediction[0])
    click = st.button("Dự đoán số")
    if click:
        st.write(f'Chữ số dự đoán là: {digit}')




    
    




    
