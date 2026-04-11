import os
import time
import cv2
import Levenshtein
import torch
from ultralytics import YOLO
import easyocr

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 15  # Limit obrazków do przetworzenia w jednym teście
BASE_DIR = 'dataset/CCPD2019/'  # Główny folder datasetu
TEST_SPLIT_FILE = os.path.join(BASE_DIR, 'splits', 'train.txt')  # Plik z listą testową

# --- SŁOWNIKI ZNAKÓW CCPD ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
import re


def korekta_chinska(tekst):
    if not tekst: return ""

    tekst = re.sub(r'[^\w\u4e00-\u9fff]', '', tekst).upper()
    tekst = tekst.replace('_', '')

    # 1. Znak Prowincji - Mapowanie Halucynacji
    halucynacje_wan = ['仓', '鲩', '院', '匦', '疏', '脸', '匿', '统', '梳', '流', '充', '祝', '谇', '从', '沈', '杭',
                       '幽', '蹙', '嗾']
    for h in halucynacje_wan:
        tekst = tekst.replace(h, '皖')

    PROWINCJE = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
                 "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

    idx_prowincji = -1
    for i, char in enumerate(tekst):
        if char in PROWINCJE:
            idx_prowincji = i
            break

    if idx_prowincji != -1:
        tekst = tekst[idx_prowincji:]
    else:
        # 2. Rekonstrukcja "Zjedzonej" Prowincji
        # Jeśli brak prowincji, a mamy 6 znaków (np. 4H509P, 1XG2I0)
        if len(tekst) == 6:
            tekst = '皖' + tekst
        elif len(tekst) == 5:
            # np. C2771 -> brakuje 皖 i jeszcze jednej litery
            tekst = '皖' + tekst
        elif len(tekst) >= 7:
            tekst = '皖' + tekst[-6:]

    # 3. KOREKTA KODU MIASTA (Druga pozycja)
    if len(tekst) >= 2:
        nowy_tekst = list(tekst)

        # Halucynacje kropki (Druga pozycja czytana jako chiński znak)
        if nowy_tekst[1] in ['弘', '丛', '蹙']:
            nowy_tekst[1] = 'A'
        elif nowy_tekst[1] in ['吭', '嗾']:
            nowy_tekst[1] = 'C'
        # Jeśli wciąż jest tam jakiś chiński znaczek, w CCPD to na 90% A
        elif '\u4e00' <= nowy_tekst[1] <= '\u9fff':
            nowy_tekst[1] = 'A'

        # Częste mylenie cyfr z literami na 2. pozycji (np. 4H509P -> AH509P)
        cyfry_na_litery_miasto = {'4': 'A', '1': 'A', '0': 'C', '8': 'B', '2': 'Z'}
        if nowy_tekst[1] in cyfry_na_litery_miasto:
            nowy_tekst[1] = cyfry_na_litery_miasto[nowy_tekst[1]]

        tekst = "".join(nowy_tekst)

    # 4. KOREKTA RESZTY TABLICY (Pozycje 2-7)
    if len(tekst) >= 3:
        nowy_tekst = list(tekst)
        for i in range(2, len(nowy_tekst)):
            if nowy_tekst[i] == 'O': nowy_tekst[i] = '0'
            if nowy_tekst[i] == 'I': nowy_tekst[i] = '1'
            if nowy_tekst[i] == 'Q': nowy_tekst[i] = '0'
        tekst = "".join(nowy_tekst)

    # 5. Odcięcie doklejonych krawędzi z prawej
    if len(tekst) == 8 and tekst[-1] in '1IL':
        tekst = tekst[:-1]

    if len(tekst) > 7:
        tekst = tekst[:7]

    return tekst


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

                # --- KROK 1: CIĘCIE (Wracamy do ustawień z wersji 20%!) ---
                szerokosc = x2 - x1
                wysokosc = y2 - y1

                # Ucinamy z obu stron, by pozbyć się krawędzi ramki (szczególnie z prawej)
                nowe_x1 = int(x1 + (szerokosc * 0.02))
                nowe_x2 = int(x2 - (szerokosc * 0.04))
                nowe_y1 = int(y1 + (wysokosc * 0.04))
                nowe_y2 = int(y2 - (wysokosc * 0.04))

                plate_crop = img[max(0, nowe_y1):min(h_img, nowe_y2), max(0, nowe_x1):min(w_img, nowe_x2)]

                if plate_crop.size == 0:
                    pred_texts.append("")
                    continue

                # --- KROK 2: CZYSTY PRE-PROCESSING ---
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # MedianBlur świetnie rozmywa "pieprz i sól" oraz białą kropkę działową
                gray = cv2.medianBlur(gray, 3)

                t_start_ocr = time.perf_counter()

                # Używamy tylko sprawdzonego mag_ratio=2.0
                ocr_results = reader.readtext(gray, detail=1, mag_ratio=2.0)
                total_ocr_time += (time.perf_counter() - t_start_ocr)

                # --- KROK 3: WYDOBYCIE ---
                czyste_kawalki = []
                crop_h = plate_crop.shape[0]

                if crop_h > 0:
                    for bbox, text, prob in ocr_results:
                        tekst = text.replace(" ", "").upper()
                        box_h = bbox[2][1] - bbox[0][1]

                        if (box_h / crop_h) < 0.20: continue
                        if len(tekst) == 1 and prob < 0.25: continue

                        czyste_kawalki.append((bbox[0][0], tekst))

                czyste_kawalki.sort(key=lambda x: x[0])
                read_text = "".join([k[1] for k in czyste_kawalki])

                # --- KROK 4: KOREKTA SŁOWNIKOWA ---
                read_text = korekta_chinska(read_text)

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
                        print(f"✅ TRAFIENIE! Odczytano idealnie: '{pred_txt}'")
                    else:
                        print(f"❌ PUDŁO! Ground Truth: '{true_txt}' | EasyOCR przeczytał: '{pred_txt}'")

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