import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import KFold

# Paths
ALL_DATA_PATH = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Load all data (train + validation combined)
all_ds = keras.preprocessing.image_dataset_from_directory(
    ALL_DATA_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    seed=42
)

# Use proper MobileNetV2 preprocessing
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# Collect all image paths and labels manually
import os

all_images = []
all_labels = []

for split in ['train', 'validation']:
    split_path = os.path.join(ALL_DATA_PATH, split, 'images')
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(split_path, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith('.jpg'):
                    all_images.append(os.path.join(class_path, img_file))
                    all_labels.append(class_idx)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f"Total images: {len(all_images)}")
print(f"Total labels: {len(all_labels)}")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f"\n=== Fold {fold + 1}/5 ===")

    train_images, val_images = all_images[train_idx], all_images[val_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

    # Build model fresh for each fold
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Create datasets for this fold
    def load_and_preprocess(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, tf.one_hot(label, NUM_CLASSES)


    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1
    )

    # Evaluate
    _, val_acc = model.evaluate(val_ds, verbose=0)
    fold_accuracies.append(val_acc)
    print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")

# Results
print("\n=== Cross-Validation Results ===")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i + 1}: {acc:.4f}")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"\nMean Accuracy: {mean_acc:.4f} ({mean_acc * 100:.2f}%)")
print(f"Std Dev: {std_acc:.4f}")
print(f"Range: {min(fold_accuracies)*100:.2f}% - {max(fold_accuracies)*100:.2f}%")