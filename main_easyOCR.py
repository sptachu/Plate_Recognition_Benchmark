import os
import time
import json
import cv2
import Levenshtein
import torch
from ultralytics import YOLO
import easyocr
import re
import numpy as np
from easyocr.model.vgg_model import Model

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 349  # Limit obrazków do przetworzenia w jednym teście
IMAGES_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu ze zdjęciami
LABELS_DIR = 'dataset/UC3M-LP/test/'  # ścieżka do folderu z plikami JSON




# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False


# --- FUNKCJE POMOCNICZE ---
class CzystyRozpoznawacz:
    def __init__(self, model_path, alfabet):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # [blank] to specjalny, pusty znak wymagany przez sieci typu CTC
        self.char_list = ['[blank]'] + list(alfabet)

        # 1. Inicjalizacja "gołej" architektury VGG (identycznej jak na treningu)
        self.model = Model(
            input_channel=1,
            output_channel=256,
            hidden_size=256,
            num_class=len(self.char_list)
        )

        # 2. Ładowanie Twoich wag
        state_dict = torch.load(model_path, map_location=self.device)

        # Oczyszczanie kluczy z prefixu 'module.' (pozostałość po treningu na GPU)
        nowe_wagi = {}
        for k, v in state_dict.items():
            nowe_wagi[k.replace('module.', '')] = v

        self.model.load_state_dict(nowe_wagi)
        self.model.to(self.device)
        self.model.eval()

    def czytaj(self, img_cv_grey):
        # 3. Sztywny resize 100x32 - dokładnie tak jak na treningu
        img = cv2.resize(img_cv_grey, (100, 32), interpolation=cv2.INTER_CUBIC)

        # 4. Normalizacja wartości pikseli do zakresu [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # 5. Tworzenie Tensora PyTorch o kształcie (Batch=1, Channel=1, H=32, W=100)
        tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
        dummy_text = torch.LongTensor(1, 25).fill_(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor, dummy_text)
            _, preds_index = preds.max(2)

        # 6. Czyste dekodowanie CTC (usuwa duplikaty i znaki [blank])
        t = preds_index[0].cpu().numpy()
        char_list = []
        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(self.char_list[t[i]])

        # Limit do 7 znaków (tablice hiszpańskie)
        wynik = ''.join(char_list)
        return wynik[:7] if len(wynik) > 7 else wynik


def inteligentny_postprocessing(tekst):
    if not tekst: return ""

    # 1. Globalne poprawki (W hiszpańskich tablicach rzadko używa się litery O, zazwyczaj to zero)
    tekst = tekst.replace('O', '0')

    # 2. Łatanie błędu CTC (zjadanie podwójnych liter na końcu)
    # Jeśli model wypluł 4 znaki (3 cyfry i 1 litera), a powinien 5, podwajamy literę!
    # Np. "505W" -> "505WW"
    if len(tekst) == 4 and tekst[:3].isdigit() and tekst[3].isalpha():
        tekst = tekst + tekst[3]

    # 3. Wymuszanie formatu (3 cyfry + 2 litery) - najczęstszy przypadek
    # Działa tylko jeśli tekst ma 5 znaków i zaczyna się od cyfry (omija formaty typu "M710P")
    if len(tekst) == 5 and (tekst[0].isdigit() or tekst[0] in 'SBZIG'):
        chars = list(tekst)

        # Słowniki pomyłek
        litery_na_cyfry = {'S': '5', 'Z': '2', 'B': '8', 'I': '1', 'G': '6', 'T': '7'}
        cyfry_na_litery = {'5': 'S', '2': 'Z', '8': 'B', '1': 'I', '0': 'D', '6': 'G'}

        # Pierwsze 3 znaki MUSZĄ być cyframi
        for i in range(3):
            if chars[i] in litery_na_cyfry:
                chars[i] = litery_na_cyfry[chars[i]]

        # Ostatnie 2 znaki MUSZĄ być literami
        for i in range(3, 5):
            if chars[i] in cyfry_na_litery:
                chars[i] = cyfry_na_litery[chars[i]]

        tekst = "".join(chars)

    return tekst

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
print("[*] Ładowanie własnego silnika PyTorch (VGG)...")
ALFABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
MODEL_PATH = './my_models/custom_uc3m.pth'
rozpoznawacz = CzystyRozpoznawacz(MODEL_PATH, ALFABET)

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

                # --- CIĘCIE IDENTYCZNE JAK W TRENINGU ---
                szerokosc = x2 - x1
                wysokosc = y2 - y1

                nowe_x1 = int(x1 + (szerokosc * 0.09))  # Odcięcie 9% z lewej
                nowe_x2 = x2  # Zostawiamy prawą stronę w spokoju
                nowe_y1 = int(y1 + (wysokosc * 0.08))  # Odcięcie 8% z góry
                nowe_y2 = int(y2 - (wysokosc * 0.08))  # Odcięcie 8% z dołu

                plate_crop = img[max(0, nowe_y1):min(h_img, nowe_y2), max(0, nowe_x1):min(w_img, nowe_x2)]

                if plate_crop.size == 0:
                    pred_texts.append("")
                    continue

                gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                # --- CZYSTY OCR PYTORCH ---
                t_start_ocr = time.perf_counter()
                read_text = rozpoznawacz.czytaj(gray_crop)
                total_ocr_time += (time.perf_counter() - t_start_ocr)

                # --- POST-PROCESSING I ZAPIS ---
                read_text = inteligentny_postprocessing(read_text)
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