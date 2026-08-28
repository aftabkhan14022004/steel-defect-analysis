import tensorflow as tf
import os

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6

# Paths
TRAIN_PATH = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\train\images"
VALID_PATH = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images"

# Check if paths exist
print("Train path exists:", os.path.exists(TRAIN_PATH))
print("Validation path exists:", os.path.exists(VALID_PATH))

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# Get class names
class_names = train_ds.class_names
print("Class names:", class_names)
print("Number of classes:", len(class_names))

# Check one batch shape
for images, labels in train_ds.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First image pixel range:", images[0].numpy().min(), "to", images[0].numpy().max())