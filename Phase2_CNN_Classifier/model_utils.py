import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import tempfile

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def get_preprocess_input():
    return keras.applications.mobilenet_v2.preprocess_input

def load_model():
    import os
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_model.h5")
    return keras.models.load_model(model_path)

def preprocess_image_from_path(image_path):
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224), color_mode='rgb', interpolation='nearest'
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = get_preprocess_input()(img_array)
    return img_array

def preprocess_image_from_array(image):
    img = image.convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = get_preprocess_input()(img_array)
    return img_array

def predict_from_path(image_path, model=None):
    if model is None:
        model = load_model()
    img_array = preprocess_image_from_path(image_path)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

def predict_from_array(image, model=None):
    if model is None:
        model = load_model()
    img_array = preprocess_image_from_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence