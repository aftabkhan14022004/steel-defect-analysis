import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import tempfile
import cv2
import mysql.connector

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


def make_gradcam_heatmap(img_array, model):
    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = base_model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=[base_model.input],
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        x = model.get_layer("global_average_pooling2d")(base_output)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("dense_1")(x)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_array = np.array(original_image.convert('RGB'))
    overlaid = np.uint8(img_array * (1 - alpha) + heatmap_colored * alpha)
    return Image.fromarray(overlaid)


def log_inspection(batch_id, image_file, actual_defect, predicted_defect, confidence, decision, line_number):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="steel_defects"
        )
        cursor = conn.cursor()
        sql = """
            INSERT INTO inspection_log 
            (batch_id, image_file, actual_defect, predicted_defect, confidence, decision, line_number, inspection_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        cursor.execute(sql, (batch_id, image_file, actual_defect, predicted_defect, confidence, decision, line_number))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Logging failed: {e}")
        return False