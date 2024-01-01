
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib

# Load the pre-trained SVM model
svm_model = joblib.load('svm_model.joblib')

# Define the label dictionary (replace with your actual labels)
label_dict = {0: 'Bacterial blight', 1: 'Brown Spot', 2: 'Copper Phytotoxicity', 3: 'Downy mildew', 4: 'Healthy',
              5: 'Powdery mildew', 6: 'Powdery mildew and Rust', 7: 'Rust and target spot', 8: 'Southern blight',
              9: 'Soybean Mosaic Virus'}

# Function to preprocess a single image
def preprocess_single_image(uploaded_file):
    # Convert the UploadedFile to a numpy array
    content = uploaded_file.read()
    nparr = np.frombuffer(content, np.uint8)

    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check if the image is loaded successfully
    if img is None:
        st.error("Error: Unable to decode the uploaded image.")
        return None

    # Resize image to 224x224
    img = cv2.resize(img, (224, 224))

    # Normalize pixel values
    img = img / 255.0

    return img

# Function to extract features from a single image using VGG16
def extract_features_single_image(img):
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    features = feature_extractor.predict(np.expand_dims(img, axis=0))
    return features

# Function to make prediction on a single image
def predict_single_image(image_path):
    # Preprocess the image
    img = preprocess_single_image(image_path)

    if img is None:
        return None

    # Extract features using VGG16
    img_features = extract_features_single_image(img)

    # Make prediction using the pre-trained SVM model
    prediction = svm_model.predict(img_features)

    # Map numerical prediction back to class label
    predicted_label = label_dict[prediction[0]]

    return predicted_label

# Streamlit app code
st.title("Leaf Disease Prediction App")

# Sidebar for image upload
st.sidebar.title("Leaf Disease Prediction")
image_path_to_predict = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_path_to_predict is not None:
    st.sidebar.image(image_path_to_predict, caption="Uploaded Image.", use_column_width=True)

    # Make prediction
    predicted_label = predict_single_image(image_path_to_predict)

    st.write("")
    st.write("## Prediction Result")
    if predicted_label is not None:
        st.success(f"The predicted label for the given image is: {predicted_label}")
    else:
        st.error("Prediction failed. Please check the uploaded image.")
