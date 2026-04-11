import os
import time
import json
import cv2
import Levenshtein
import torch
from ultralytics import YOLO
import easyocr
import re
# --- USTAWIENIA TESTU ---
MAX_IMAGES = 15  # Limit obrazków do przetworzenia w jednym teście
IMAGES_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu ze zdjęciami
LABELS_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu z plikami JSON




# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False


# --- FUNKCJE POMOCNICZE ---


def korekta_hiszpanska(tekst):
    if not tekst: return ""
    tekst = tekst.strip('E').strip('F').replace(' ', '').upper()

    # Usuwamy halucynacje krawędzi
    if len(tekst) == 6 and tekst[0] in '107':
        tekst = tekst[1:]
    if len(tekst) > 5:
        tekst = tekst[-5:]

    # KOREKTA POZYCYJNA (3 cyfry, 2 litery)
    if len(tekst) == 5:
        nowy_tekst = list(tekst)
        litery_na_cyfry = {'O': '0', 'Q': '0', 'I': '1', 'A': '4', 'L': '4', 'S': '5', 'Z': '2', 'B': '8', 'G': '6'}
        cyfry_na_litery = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '4': 'A'}

        # Poprawiamy tylko nowe formaty hiszpańskie
        if not (nowy_tekst[0].isalpha() and nowy_tekst[1].isdigit()):
            for i in range(3):
                if nowy_tekst[i] in litery_na_cyfry:
                    nowy_tekst[i] = litery_na_cyfry[nowy_tekst[i]]
            for i in range(3, 5):
                if nowy_tekst[i] in cyfry_na_litery:
                    nowy_tekst[i] = cyfry_na_litery[nowy_tekst[i]]

        tekst = "".join(nowy_tekst)

    return tekst

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
print("[*] Ładowanie EasyOCR...")
reader = easyocr.Reader(['en'], gpu=use_gpu_ocr)

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
                # --- POCZĄTEK PĘTLI: Pobranie koordynatów z YOLO ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # [ZASADA 1]: Zawsze dodajemy oryginalną ramkę do statystyk (chroni przed IndexError)
                pred_boxes.append([x1, y1, x2, y2])

                # --- KROK 3: OSTRZEJSZE CIĘCIE (Zabijamy 'E' i ramki) ---
                szerokosc = x2 - x1
                wysokosc = y2 - y1

                # Ucinamy aż 8% z lewej, żeby ostatecznie pożegnać się z literką 'E' z flagi UE
                nowe_x1 = int(x1 + (szerokosc * 0.09))
                nowe_x2 = int(x2 - (szerokosc * 0.02))
                # Tniemy mocniej z góry i z dołu, żeby zabić napisy dealerskie pod tablicą
                nowe_y1 = int(y1 + (wysokosc * 0.08))
                nowe_y2 = int(y2 - (wysokosc * 0.08))

                plate_crop = img[max(0, nowe_y1):min(h_img, nowe_y2), max(0, nowe_x1):min(w_img, nowe_x2)]

                # Jeśli ucięliśmy za dużo i obrazek jest pusty, oddajemy pusty string
                if plate_crop.size == 0:
                    pred_texts.append("")
                    continue

                # --- KROK 4: OTSU BINARYZACJA (Czarno-biały skan) ---
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # Magia OCR: Otsu automatycznie znajduje idealny próg, żeby oddzielić czarne litery od białego tła
                _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                t_start_ocr = time.perf_counter()
                dozwolone_znaki = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

                # Puszczamy idealnie czarno-biały obraz do modelu
                ocr_results = reader.readtext(binary_img, detail=1, allowlist=dozwolone_znaki)
                total_ocr_time += (time.perf_counter() - t_start_ocr)

                # --- KROK 5: RYGORYSTYCZNY FILTR GEOMETRYCZNY ---
                czyste_kawalki = []
                crop_h, crop_w = plate_crop.shape[:2]

                if crop_h > 0:
                    for bbox, text, prob in ocr_results:
                        tekst = text.replace(" ", "").upper()
                        box_h = bbox[2][1] - bbox[0][1]

                        # 1. Główny filtr wysokości (Zabójca śrubek):
                        # Litery na tablicy są wysokie. Jeśli znak zajmuje mniej niż 40% wysokości obrazka - to śmieć!
                        if (box_h / crop_h) < 0.40:
                            continue

                        # 2. Filtr halucynacji: EasyOCR często zmyśla pojedyncze znaki ("L", "1").
                        # Jeśli wykrył tylko 1 znak, musi być go pewien na minimum 60%.
                        if len(tekst) == 1 and prob < 0.60:
                            continue

                        # 3. Jeśli przeszło powyższe testy, bierzemy do wyniku
                        if len(tekst) >= 1 and prob > 0.35:
                            czyste_kawalki.append((bbox[0][0], tekst))

                czyste_kawalki.sort(key=lambda x: x[0])
                read_text = "".join([k[1] for k in czyste_kawalki])

                if len(read_text) > 7:
                    read_text = read_text[:7]

                    # --- KROK 6: KOŁO RATUNKOWE (FALLBACK OCR) ---
                    # Jeśli po naszych rygorystycznych filtrach zostało nam nic, albo tylko 1-2 litery (podejrzenie błędu)
                if len(read_text) < 3:
                    # 1. Cofamy się do oryginalnej ramki YOLO z minimalnym, kosmetycznym cięciem 2%
                    nowe_x1_fb = int(x1 + (szerokosc * 0.02))
                    nowe_x2_fb = int(x2 - (szerokosc * 0.02))
                    nowe_y1_fb = int(y1 + (wysokosc * 0.02))
                    nowe_y2_fb = int(y2 - (wysokosc * 0.02))

                    plate_crop_fb = img[max(0, nowe_y1_fb):min(h_img, nowe_y2_fb),
                                    max(0, nowe_x1_fb):min(w_img, nowe_x2_fb)]

                    if plate_crop_fb.size > 0:
                        # 2. Rezygnujemy z Blura (mógł stopić małe litery)
                        gray_fb = cv2.cvtColor(plate_crop_fb, cv2.COLOR_BGR2GRAY)
                        gray_fb = cv2.normalize(gray_fb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                        t_start_fb = time.perf_counter()
                        # 3. Odpalamy z mag_ratio=1.5 (powiększenie, które ratuje zamazane tablice)
                        fb_results = reader.readtext(gray_fb, detail=1, allowlist=dozwolone_znaki, mag_ratio=1.5)
                        total_ocr_time += (time.perf_counter() - t_start_fb)

                        fb_kawalki = []
                        crop_h_fb = plate_crop_fb.shape[0]

                        for bbox, text, prob in fb_results:
                            tekst = text.replace(" ", "").upper()
                            box_h = bbox[2][1] - bbox[0][1]

                            # Bardzo łagodne filtry dla koła ratunkowego (przepuszczą niemal wszystko, co ma sens)
                            if (box_h / crop_h_fb) < 0.15:
                                continue
                            if len(tekst) >= 1 and prob > 0.25:
                                fb_kawalki.append((bbox[0][0], tekst))

                        fb_kawalki.sort(key=lambda x: x[0])
                        read_text = "".join([k[1] for k in fb_kawalki])

                        if len(read_text) > 7:
                            read_text = read_text[:7]

                    # Koniec koła ratunkowego, odkładamy ostateczny wynik
                read_text = korekta_hiszpanska(read_text)
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