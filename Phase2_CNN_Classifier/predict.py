from model_utils import predict_from_path

test_images = [
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\crazing\crazing_241.jpg",
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\inclusion\inclusion_241.jpg",
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\patches\patches_241.jpg",
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\pitted_surface\pitted_surface_241.jpg",
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\rolled-in_scale\rolled-in_scale_241.jpg",
    r"C:\Users\aftab\Desktop\Steel_project\data\raw\NEU-DET\validation\images\scratches\scratches_241.jpg"
]

for img_path in test_images:
    predicted, confidence = predict_from_path(img_path)
    actual = img_path.split("\\")[-2]
    filename = img_path.split("\\")[-1]
    print(f"{filename}: {actual} → {predicted} ({confidence:.2f}%)")