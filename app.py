import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Fruit and Veg Classifier", page_icon="🍏", layout="wide")

# Load the pre-trained model
model = load_model('Fruits and Vegetable Classifier.keras')

# List of categories (class names)
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

# Define image dimensions
img_width = 180
img_height = 180

# Sidebar for uploading the image
st.sidebar.header('Upload an Image')
image = st.sidebar.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

# Main header
st.title('Fruits and Vegetables Image Classification CNN')
st.write('Upload an image of a fruit or vegetable and the model will classify it.')

# Function to preprocess and predict the uploaded image
@st.cache_data
def predict_image(image):
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])
    
    # Get the highest confidence category
    predicted_label = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    
    return predicted_label, confidence

# Display the image and prediction result
if image is not None:
    st.image(image, caption='Uploaded Image', width=350)
    predicted_label, confidence = predict_image(image)
    
    # Use columns for layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Prediction')
        st.write(f'**{predicted_label}**')
    with col2:
        st.subheader('Confidence')
        st.write(f'**{confidence:.2f}%**')

    st.success('Classification Complete!')
else:
    st.info('Please upload an image file to classify.')

# Additional information in the sidebar
st.sidebar.markdown('''
This application uses a Convolutional Neural Network (CNN) to classify images of fruits and vegetables. 
Upload an image to see the classification result.
''')

# Footer or additional information
st.sidebar.markdown('''
---
**Note:** The model is trained to classify images into one of the predefined categories.
''')
