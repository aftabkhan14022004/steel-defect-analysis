import tensorflow as tf
from tensorflow import keras
import numpy as np

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

model = keras.models.load_model('best_model.h5')


def predict_defect(image_path):
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224), color_mode='rgb'
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence


# Test on a sample image
test_image = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\crazing\crazing_241.jpg"
predicted, conf = predict_defect(test_image)
print(f"Predicted: {predicted}, Confidence: {conf:.2f}%")