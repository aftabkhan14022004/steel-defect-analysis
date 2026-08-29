import os
from model_utils import CLASS_NAMES, load_model, predict_from_path

VALID_PATH = r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images"

model = load_model()

misclassified = []
correct = 0
total = 0

for class_name in CLASS_NAMES:
    class_path = os.path.join(VALID_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_file in os.listdir(class_path):
        if not img_file.endswith('.jpg'):
            continue

        img_path = os.path.join(class_path, img_file)
        predicted, confidence = predict_from_path(img_path, model)

        total += 1
        if predicted == class_name:
            correct += 1
        else:
            misclassified.append({
                'file': img_file,
                'actual': class_name,
                'predicted': predicted,
                'confidence': confidence
            })

accuracy = (correct / total) * 100

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Misclassified: {len(misclassified)}")
print(f"Accuracy: {accuracy:.2f}%")

print("\n=== Misclassified Images ===")
for item in misclassified:
    print(f"{item['file']}: {item['actual']} → {item['predicted']} ({item['confidence']:.2f}%)")