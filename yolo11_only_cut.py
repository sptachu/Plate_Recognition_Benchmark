import os
import cv2
import json
import torch
from ultralytics import YOLO

# --- USTAWIENIA ---
BASE_DIR = 'dataset/UC3M-LP/'
# Definiujemy, które foldery chcemy przetworzyć
SPLITS = ['train', 'test']


def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False


def oblicz_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def czytaj_ground_truth_json(json_path):
    gt_boxes = []
    gt_texts = []
    if not os.path.exists(json_path): return gt_boxes, gt_texts

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for lp in data.get('lps', []):
        poly = lp.get('poly_coord', [])
        if len(poly) == 4:
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            gt_boxes.append([x1, y1, x2, y2])
        else:
            continue

        chars = lp.get('characters', [])
        prawdziwy_tekst = "".join([c.get('char_id', '') for c in chars]).replace(" ", "").upper()
        gt_texts.append(prawdziwy_tekst)

    return gt_boxes, gt_texts


def wytnij_i_zapisz():
    device_yolo, _ = detect_hardware()
    print(f"[*] Ładowanie YOLO11 na urządzeniu: {device_yolo.upper()}...")
    detector = YOLO('yolo11_plate.pt')

    for split in SPLITS:
        input_dir = os.path.join(BASE_DIR, split)
        output_dir = os.path.join(BASE_DIR, f"{split}_ocr")

        if not os.path.exists(input_dir):
            print(f"[!] Pomijam {split}, brak folderu: {input_dir}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        labels_file_path = os.path.join(output_dir, 'labels.txt')
        log_file_path = os.path.join(output_dir, '.processed_files.txt')

        # --- SYSTEM WZNAWIANIA (RESUME) ---
        przetworzone_pliki = set()
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                przetworzone_pliki = set(line.strip() for line in f if line.strip())

        saved_crops_count = 0
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'r', encoding='utf-8') as f:
                saved_crops_count = sum(1 for _ in f)

        print(f"\n[*] Przetwarzanie zbioru: {split.upper()}")
        print(f"[*] Zapis do: {output_dir}")

        if len(przetworzone_pliki) > 0:
            print(
                f"[+] Znaleziono postęp! Pominę {len(przetworzone_pliki)} obrazków i zacznę od tablicy nr {saved_crops_count}.")

        processed_count = len(przetworzone_pliki)

        # Otwieramy pliki w trybie 'a' (append) żeby nie usunąć starych danych
        with open(labels_file_path, 'a', encoding='utf-8') as labels_file, \
                open(log_file_path, 'a', encoding='utf-8') as log_file:

            for filename in sorted(os.listdir(input_dir)):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Jeśli plik już był zrobiony, omijamy go
                if filename in przetworzone_pliki:
                    continue

                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(input_dir, filename)
                json_path = os.path.join(input_dir, base_name + '.json')

                img = cv2.imread(img_path)
                if img is None:
                    continue

                h_img, w_img, _ = img.shape
                gt_boxes, gt_texts = czytaj_ground_truth_json(json_path)

                if gt_boxes:
                    # Detekcja
                    results = detector(img, device=device_yolo, verbose=False)
                    pred_boxes = []

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            pred_boxes.append([x1, y1, x2, y2])

                    # Parowanie wykrytych ramek z JSONem za pomocą IoU
                    matched_gt = set()
                    for p_idx, p_box in enumerate(pred_boxes):
                        best_iou, best_gt_idx = 0.5, -1
                        for g_idx, g_box in enumerate(gt_boxes):
                            if g_idx in matched_gt: continue
                            iou = oblicz_iou(p_box, g_box)
                            if iou >= best_iou:
                                best_iou, best_gt_idx = iou, g_idx

                        if best_gt_idx != -1:
                            matched_gt.add(best_gt_idx)
                            true_txt = gt_texts[best_gt_idx]

                            x1, y1, x2, y2 = p_box
                            szerokosc = x2 - x1
                            wysokosc = y2 - y1

                            nowe_x1 = int(x1 + (szerokosc * 0.09))
                            nowe_x2 = x2
                            nowe_y1 = int(y1 + (wysokosc * 0.08))
                            nowe_y2 = int(y2 - (wysokosc * 0.08))

                            plate_crop = img[max(0, nowe_y1):min(h_img, nowe_y2), max(0, nowe_x1):min(w_img, nowe_x2)]

                            if plate_crop.size > 0:
                                crop_filename = f"{split}_crop_{saved_crops_count:06d}.jpg"
                                crop_filepath = os.path.join(output_dir, crop_filename)

                                cv2.imwrite(crop_filepath, plate_crop)
                                labels_file.write(f"{crop_filename}\t{true_txt}\n")
                                saved_crops_count += 1

                # Zapisujemy plik jako zrobiony (nawet jeśli nie było na nim tablic)
                log_file.write(filename + '\n')
                log_file.flush()  # Wymusza zapis na dysk w razie zatrzymania skryptu

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"    Przetworzono oryginalnych obrazków: {processed_count}", end='\r')

        print(f"\n[+] Zakończono zbiór {split}. Aktualnie w bazie znajduje się {saved_crops_count} wyciętych tablic.")


if __name__ == "__main__":
    wytnij_i_zapisz()