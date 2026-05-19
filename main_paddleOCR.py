import os
import torch
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
import time
import json
import cv2
import Levenshtein
import re
from ultralytics import YOLO

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 400  # Limit obrazków do przetworzenia w jednym teście
IMAGES_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu ze zdjęciami
LABELS_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu z plikami JSON


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


def czytaj_ground_truth_json(json_path):
    """Odczytuje pliki JSON z bazy UC3M-LP."""
    gt_boxes = []
    gt_texts = []

    if not os.path.exists(json_path):
        return gt_boxes, gt_texts

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for lp in data.get('lps', []):
        # 1. Wyciąganie koordynatów i konwersja wielokąta na prostokąt (Bounding Box)
        poly = lp.get('poly_coord', [])
        if len(poly) == 4:
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            gt_boxes.append([x1, y1, x2, y2])
        else:
            continue

        # 2. Składanie prawdziwego tekstu ze znaków
        chars = lp.get('characters', [])
        prawdziwy_tekst = "".join([c.get('char_id', '') for c in chars])
        prawdziwy_tekst = prawdziwy_tekst.replace(" ", "").upper()
        gt_texts.append(prawdziwy_tekst)

    return gt_boxes, gt_texts


# --- INICJALIZACJA MODELI ---
device_yolo, use_gpu_ocr = detect_hardware()
print("[*] Ładowanie YOLO11...")
detector = YOLO('yolo11_plate.pt')
print("[*] Ładowanie PaddleOCR...")
# det=False disables Paddle's built-in text detection (YOLO does this for us)
# lang='en' works best for standard alphanumeric license plates

# 2. Use the updated argument names
reader = PaddleOCR(
    rec_model_dir="./inference_model/",
    rec_char_dict_path="./lp_dict.txt",
    use_angle_cls=False,
    use_gpu=False
)

# --- ZMIENNE DO STATYSTYK ---
TP, FP, FN = 0, 0, 0
exact_matches = 0
total_cer = 0.0
ocr_evaluated_count = 0

total_yolo_time, total_ocr_time, total_e2e_time = 0.0, 0.0, 0.0
processed_images = 0

print("\n[*] Rozpoczynam ewaluację na datasecie UC3M-LP...\n")

try:
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Parowanie zdjęcia z odpowiednim plikiem JSON
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGES_DIR, filename)
        json_path = os.path.join(LABELS_DIR, base_name + '.json')

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img, _ = img.shape
        gt_boxes, gt_texts = czytaj_ground_truth_json(json_path)

        # Ominięcie zdjęcia, jeśli Ground Truth jest pusty
        if not gt_boxes:
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

                pad = -15 # pixels
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # width = x2 - x1
                # height = y2 - y1

                # crop_x1 = int(x1 + (width * 0.09))
                # crop_y1 = max(0, y1 - pad)
                # crop_x2 = min(w_img, x2 + pad)
                # crop_y2 = min(h_img, y2 + pad)

                # plate_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                plate_crop = img[y1:y2, x1:x2]
                plate_crop = cv2.resize(plate_crop, (320, 48))


                if plate_crop.size == 0:
                    pred_texts.append("")
                    continue
                
                t_start_ocr = time.perf_counter()
                
                # Pass the crop directly to PaddleOCR. 
                # cls=False disables orientation classification to save time.
                ocr_results = reader.ocr(plate_crop, det=False, cls=False)
                total_ocr_time += (time.perf_counter() - t_start_ocr)

                raw_text = ""
                # Safely extract text: check if results exist and aren't None
                if ocr_results and isinstance(ocr_results[0], list):
                    # ocr_results[0][0] is (text, confidence)
                    raw_text = ocr_results[0][0][0]

                raw_text = raw_text.replace(" ", "").upper()
                read_text = raw_text[:5]
                
                # read_text = "".join(ocr_results).replace(" ", "").upper()
                pred_texts.append(read_text)

        total_e2e_time += (time.perf_counter() - t_start_e2e)

        # # 1. First, make sure all drawing happens BEFORE showing the image
        # for p_idx, p_box in enumerate(pred_boxes):
        #     px1, py1, px2, py2 = p_box
        #     width = x2 - x1
        #     height = y2 - y1

        #     px1 = int(x1 + (width * 0.09))
        #     py1 = max(0, y1 - pad)
        #     px2 = min(w_img, x2 + pad)
        #     py2 = min(h_img, y2 + pad)
        #     text = pred_texts[p_idx]
            
        #     # Draw the box and the text on the 'img'
        #     cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 3)
        #     cv2.putText(img, text, (px1, py1 - 15), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # # 2. Scale the image down so it fits on your monitor
        # # We calculate a scaling factor (e.g., 0.5 = half size)
        # screen_res = 1280, 720  # Target resolution for your monitor
        # scale_width = screen_res[0] / img.shape[1]
        # scale_height = screen_res[1] / img.shape[0]
        # scale = min(scale_width, scale_height)

        # # Only scale down if the image is actually bigger than the screen
        # if scale < 1.0:
        #     window_w = int(img.shape[1] * scale)
        #     window_h = int(img.shape[0] * scale)
        #     display_img = cv2.resize(img, (window_w, window_h))
        # else:
        #     display_img = img

        # # 3. Show the scaled image
        # cv2.imshow("Plate Benchmark - Press any key for next", display_img)

        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'): # Press 'q' to quit early
        #     break

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
                    edit_dist = Levenshtein.distance(true_txt, pred_txt)
                    if true_txt == pred_txt:
                        exact_matches += 1
                        print(f"✅ TRAFIENIE! Odczytano idealnie: '{pred_txt}'")
                    else:
                        print(f"❌ PUDŁO! Ground Truth: '{true_txt}' | PaddleOCR przeczytał: '{pred_txt}' | Odległość Levenshteina: {edit_dist}")

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
        print("\n--- [2] Trafność Rozpoznawania (PaddleOCR) ---")
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

        # Znajdowanie najwyższego numeru
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

        # Zapis prostych danych klucz:wartość (bardzo łatwe do odczytu przez drugi skrypt)
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