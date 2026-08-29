from tensorflow import keras

model = keras.models.load_model(r"C:\Users\aftab\Desktop\Steel_project\best_model.h5")
model.save("best_model.keras")
print("Model converted to Keras v3 format")