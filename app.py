#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# Load the trained model and the label encoder
model = tf.keras.models.load_model('iris_nn_model.h5')
encoder = joblib.load('label_encoder.pkl')

# Load the Iris dataset to get class names
from sklearn.datasets import load_iris
iris = load_iris()

# Streamlit app title
st.title("Iris Flower Species Prediction")

# Create form for user input
st.header("Input features")
sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 0.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.0, 5.0, 1.5)

# Make predictions
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})
prediction = model.predict(input_data)
predicted_class = encoder.inverse_transform(prediction)[0]

# Display the result
st.header("Prediction")
st.write(f"The predicted class is: {iris.target_names[predicted_class]}")
st.write(f"Prediction probabilities: {prediction}")

# Show the class labels
st.header("Class labels")
st.write({i: name for i, name in enumerate(iris.target_names)})

