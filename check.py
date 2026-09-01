from tensorflow import keras
model = keras.models.load_model('best_model.h5')
base_model = model.layers[0]
print(base_model.name)
for layer in base_model.layers[-6:]:
    print(layer.name, layer.output_shape)