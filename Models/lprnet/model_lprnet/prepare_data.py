from ultralytics import YOLO
import cv2
import os

model_yolo = YOLO('yolo11_plate.pt')
input_dir = 'dataset/train_full_cars/'
output_dir = 'dataset/lprnet_train_crops/'

for img_name in os.listdir(input_dir):
    if not img_name.endswith('.jpg'): continue

    img = cv2.imread(os.path.join(input_dir, img_name))
    results = model_yolo(img)[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        # Przykład: nazwa pliku to label z Twojego datasetu
        # label = pobierz_label_dla_tego_pliku(img_name)
        # cv2.imwrite(f"{output_dir}/{label}_{i}.jpg", crop)