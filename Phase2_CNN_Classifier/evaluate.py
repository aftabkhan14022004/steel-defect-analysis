import os
import numpy as np
from model_utils import CLASS_NAMES, load_model, predict_from_path

VALID_PATH = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images"

model = load_model()

y_true = []
y_pred = []

for class_name in CLASS_NAMES:
    class_path = os.path.join(VALID_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_file in os.listdir(class_path):
        if not img_file.endswith('.jpg'):
            continue

        img_path = os.path.join(class_path, img_file)
        predicted, _ = predict_from_path(img_path, model)

        y_true.append(class_name)
        y_pred.append(predicted)

from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")