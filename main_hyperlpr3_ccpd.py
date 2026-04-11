import os
import time
import cv2
import Levenshtein
import re
import hyperlpr3 as lpr3

# --- KONFIGURACJA CCPD ---
MAX_IMAGES = 1000
IMAGES_DIR = 'dataset/CCPD2019/ccpd_base/'  # Zmień na swoją ścieżkę

# Mapowanie indeksów CCPD na znaki
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
             "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def decode_ccpd_filename(filename):
    """
    Format CCPD:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-110-72.jpg
    Części:
    [2]: Box (154&383_386&473)
    [4]: Numer tablicy (0_0_22_27_27_33_16)
    """
    try:
        name = os.path.splitext(filename)[0]
        parts = name.split('-')

        # 1. Dekodowanie Bounding Boxa
        box_coords = parts[2].split('_')
        x1y1 = box_coords[0].split('&')
        x2y2 = box_coords[1].split('&')
        gt_box = [int(x1y1[0]), int(x1y1[1]), int(x2y2[0]), int(x2y2[1])]

        # 2. Dekodowanie Tekstu
        indices = parts[4].split('_')
        # Prowincja (indeks 0) + Litera (indeks 1) + reszta znaków
        res_text = provinces[int(indices[0])] + alphabets[int(indices[1])]
        for i in indices[2:]:
            res_text += alphabets[int(i)]

        return gt_box, res_text
    except Exception as e:
        return None, None


def oblicz_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# --- INICJALIZACJA ---
print("[*] Inicjalizacja HyperLPR3 (CCPD Mode)...")
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

TP, FP, FN = 0, 0, 0
exact_matches, total_cer, processed_images = 0, 0.0, 0
total_time = 0.0

print(f"[*] Rozpoczynam test na {MAX_IMAGES} obrazach z bazy CCPD...\n")

files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in files:
    img_path = os.path.join(IMAGES_DIR, filename)
    gt_box, gt_text = decode_ccpd_filename(filename)

    if gt_box is None: continue

    img = cv2.imread(img_path)
    if img is None: continue

    t_start = time.perf_counter()
    results = catcher(img)
    total_time += (time.perf_counter() - t_start)

    found_this_img = False
    for res_text, conf, p_type, p_box in results:
        iou = oblicz_iou(p_box, gt_box)

        # Jeśli detekcja pokrywa się z Ground Truth (IoU > 0.45)
        if iou > 0.45:
            TP += 1
            found_this_img = True

            # W CCPD HyperLPR3 powinien czytać znaki chińskie, więc nie czyścimy ich!
            # Usuwamy tylko ewentualne spacje
            pred_text = res_text.replace(" ", "")

            is_match = (gt_text == pred_text)
            if is_match: exact_matches += 1

            cer = Levenshtein.distance(gt_text, pred_text) / len(gt_text)
            total_cer += cer

            status = "✅" if is_match else "❌"
            print(f"{status} GT: {gt_text} | Pred: {pred_text} | IoU: {iou:.2f} | CER: {cer:.2f}")
            break  # Ewaluujemy tylko jedną tablicę na zdjęcie w CCPD

    if not found_this_img:
        FN += 1
        print(f"⚪ NIE WYKRYTO: {gt_text}")

    processed_images += 1
    if processed_images >= MAX_IMAGES: break

# --- RAPORT ---
if processed_images > 0:
    acc = (exact_matches / TP * 100) if TP > 0 else 0
    avg_cer = (total_cer / TP) if TP > 0 else 1.0
    print("\n" + "=" * 50)
    print(f"WYNIKI CCPD (HyperLPR3)")
    print(f"Przetworzono: {processed_images} zdjęć")
    print(f"Detekcja TP: {TP} | FN: {FN}")
    print(f"OCR Accuracy: {acc:.2f}%")
    print(f"Średni CER: {avg_cer:.4f}")
    print(f"Średni czas: {(total_time / processed_images) * 1000:.2f} ms")
    print("=" * 50)