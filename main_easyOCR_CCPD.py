import os
import time
import cv2
import Levenshtein
import torch
from ultralytics import YOLO
import easyocr

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 15  # Limit obrazków do przetworzenia w jednym teście
BASE_DIR = 'dataset/CCPD2019/CCPD2019/'  # Główny folder datasetu
TEST_SPLIT_FILE = os.path.join(BASE_DIR, 'splits', 'train.txt') # Plik z listą testową

# --- SŁOWNIKI ZNAKÓW CCPD ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False

# --- FUNKCJE POMOCNICZE ---
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

def parse_ccpd_filename(filename):
    """Dekoduje Ground Truth z nazwy pliku CCPD."""
    # Usuwamy ścieżki i rozszerzenie, zostawiamy samą nazwę
    base_name = os.path.basename(filename).replace('.jpg', '').replace('.png', '')
    parts = base_name.split('-')
    
    # Jeśli format nazwy jest nieprawidłowy, pomiń
    if len(parts) < 7:
        return [], []
        
    # 1. Wyciąganie Bounding Boxa (część 2 w nazwie)
    # Przykład: "154&383_386&473"
    bbox_part = parts[2].split('_')
    p1 = bbox_part[0].split('&')
    p2 = bbox_part[1].split('&')
    
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    
    # Upewniamy się, że format to [min_x, min_y, max_x, max_y]
    gt_box = [[min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]]
    
    # 2. Składanie prawdziwego tekstu (część 4 w nazwie)
    # Przykład: "0_0_22_27_27_33_16"
    indices = parts[4].split('_')
    text = ""
    if len(indices) == 7:
        text += provinces[int(indices[0])]
        text += alphabets[int(indices[1])]
        for i in range(2, 7):
            text += ads[int(indices[i])]
            
    # "O" oznacza brak znaku, więc go usuwamy
    text = text.replace("O", "")
    
    return gt_box, [text]


# --- INICJALIZACJA MODELI ---
device_yolo, use_gpu_ocr = detect_hardware()
print("[*] Ładowanie YOLO11...")
detector = YOLO('yolo11_plate.pt')

print("[*] Ładowanie EasyOCR...")
# UWAGA: Dodano 'ch_sim', aby czytać chińskie znaki prowincji!
reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu_ocr)

# --- ZMIENNE DO STATYSTYK ---
TP, FP, FN = 0, 0, 0
exact_matches = 0
total_cer = 0.0
ocr_evaluated_count = 0

total_yolo_time, total_ocr_time, total_e2e_time = 0.0, 0.0, 0.0
processed_images = 0

print("\n[*] Rozpoczynam ewaluację na datasecie CCPD2019...\n")

try:
    # Pobieranie listy obrazków z pliku test.txt
    if not os.path.exists(TEST_SPLIT_FILE):
        raise FileNotFoundError(f"Nie znaleziono pliku splitu: {TEST_SPLIT_FILE}")
        
    with open(TEST_SPLIT_FILE, 'r') as f:
        test_images = [line.strip() for line in f.readlines() if line.strip()]

    for rel_path in test_images:
        img_path = os.path.join(BASE_DIR, rel_path)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Nie udało się wczytać obrazu: {img_path}")
            continue

        h_img, w_img, _ = img.shape
        gt_boxes, gt_texts = parse_ccpd_filename(rel_path)

        # Ominięcie zdjęcia, jeśli Ground Truth jest pusty (zła nazwa pliku)
        if not gt_boxes or not gt_texts[0]:
            continue

        t_start_e2e = time.perf_counter()

        # Detekcja YOLO
        t_start_yolo = time.perf_counter()
        results = detector(img, device=device_yolo, verbose=False)
        total_yolo_time += (time.perf_counter() - t_start_yolo)

        pred_boxes, pred_texts = [], []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pred_boxes.append([x1, y1, x2, y2])

                plate_crop = img[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

                if plate_crop.size == 0:
                    pred_texts.append("")
                    continue

                t_start_ocr = time.perf_counter()
                ocr_results = reader.readtext(plate_crop, detail=0)
                total_ocr_time += (time.perf_counter() - t_start_ocr)

                read_text = "".join(ocr_results).replace(" ", "").upper()
                pred_texts.append(read_text)

        total_e2e_time += (time.perf_counter() - t_start_e2e)

        # Ewaluacja Detekcji i OCR
        matched_gt = set()
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou, best_gt_idx = 0.5, -1

            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt: continue
                iou = oblicz_iou(p_box, g_box)
                if iou >= best_iou:
                    best_iou, best_gt_idx = iou, g_idx

            if best_gt_idx != -1:
                TP += 1
                matched_gt.add(best_gt_idx)

                # Ocena OCR
                true_txt = gt_texts[best_gt_idx]
                pred_txt = pred_texts[p_idx]

                if true_txt:
                    ocr_evaluated_count += 1
                    if true_txt == pred_txt:
                        exact_matches += 1
                    else:
                        print(f"PUDŁO! Ground Truth: '{true_txt}' | EasyOCR przeczytał: '{pred_txt}'")

                    edit_dist = Levenshtein.distance(true_txt, pred_txt)
                    total_cer += edit_dist / len(true_txt) if len(true_txt) > 0 else 1.0

            else:
                FP += 1

        FN += (len(gt_boxes) - len(matched_gt))
        processed_images += 1

        print(f"Przetworzono obrazków: {processed_images}/{MAX_IMAGES}", end='\r')

        if processed_images >= MAX_IMAGES:
            print(f"\n[!] Osiągnięto limit {MAX_IMAGES} obrazków. Kończę testy.")
            break

except KeyboardInterrupt:
    print("\n[!] Przerwano ręcznie przez użytkownika.")

finally:
    if processed_images > 0:
        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0.0

        avg_yolo_ms = (total_yolo_time / processed_images) * 1000
        avg_ocr_ms = (total_ocr_time / processed_images) * 1000
        avg_e2e_ms = (total_e2e_time / processed_images) * 1000

        plate_acc = (exact_matches / ocr_evaluated_count * 100) if ocr_evaluated_count > 0 else 0.0
        avg_cer = (total_cer / ocr_evaluated_count) if ocr_evaluated_count > 0 else 0.0

        # --- WYŚWIETLENIE RAPORTU W KONSOLI ---
        print("\n==================================================")
        print("                RAPORT Z EWALUACJI")
        print("==================================================")
        print(f"Przeanalizowane obrazy: {processed_images}")
        print("\n--- [1] Metryki Detekcji (YOLO11) ---")
        print(f"Precyzja (Precision): {P:.4f}")
        print(f"Czułość (Recall):     {R:.4f}")
        print(f"F1-Score:             {F1:.4f}")
        print(f"   (TP: {TP}, FP: {FP}, FN: {FN})")
        print("\n--- [2] Trafność Rozpoznawania (EasyOCR) ---")
        print(f"Plate-level Accuracy (Exact Match): {plate_acc:.2f}%")
        print(f"Character Error Rate (CER):         {avg_cer:.4f}")
        print("\n--- [3] Wydajność (Latency) ---")
        print(f"Średni czas inferencji detektora:   {avg_yolo_ms:.2f} ms / obraz")
        print(f"Średni czas inferencji OCR:         {avg_ocr_ms:.2f} ms / obraz")
        print(f"Średnie opóźnienie End-to-End:      {avg_e2e_ms:.2f} ms / obraz")
        print("==================================================")

        # --- ZAPIS WYNIKÓW DO PLIKU TXT ---
        RESULTS_DIR = "results"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        istniejace = [f for f in os.listdir(RESULTS_DIR) if f.startswith("results_yolo11+ocr_") and f.endswith(".txt")]
        nr = 1
        if istniejace:
            numery = []
            for f in istniejace:
                try:
                    numery.append(int(f.split('_')[-1].split('.')[0]))
                except ValueError:
                    pass
            if numery:
                nr = max(numery) + 1

        nazwa_pliku = f"results_yolo11+ocr_{nr}.txt"
        sciezka_zapisu = os.path.join(RESULTS_DIR, nazwa_pliku)

        with open(sciezka_zapisu, 'w', encoding='utf-8') as f:
            f.write(f"TP:{TP}\n")
            f.write(f"FP:{FP}\n")
            f.write(f"FN:{FN}\n")
            f.write(f"Precision:{P}\n")
            f.write(f"Recall:{R}\n")
            f.write(f"F1:{F1}\n")
            f.write(f"Plate_Accuracy:{plate_acc}\n")
            f.write(f"CER:{avg_cer}\n")
            f.write(f"YOLO_ms:{avg_yolo_ms}\n")
            f.write(f"OCR_ms:{avg_ocr_ms}\n")
            f.write(f"E2E_ms:{avg_e2e_ms}\n")

        print(f"[+] Zapisano pomyślnie wyniki do pliku: {sciezka_zapisu}")