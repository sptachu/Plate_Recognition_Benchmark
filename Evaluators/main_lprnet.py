import os
import time
import cv2
import torch
import numpy as np
import json
import re
import Levenshtein
from ultralytics import YOLO
from Models.lprnet.model_lprnet.LPRNet import build_lprnet
from Tools.load_data import CHARS

# --- KONFIGURACJA ---
YOLO_MODEL = 'yolo11_plate.pt'
LPR_MODEL = './Models/lprnet/weights_lprnet/europe/LPRNet_Euro_Epoch_100.pth'
TEST_IMAGES_DIR = './dataset/UC3M-LP/test/'  # Folder ze zdjęciami i JSONami
RESULTS_DIR = "results"
IMG_SIZE = [94, 24]


def wazony_levenshtein(s1, s2):
    """
    Profesjonalne wagi dla systemów OCR na tablicach rejestracyjnych.
    """
    kary = {
        ('0', 'O'): 0.1, ('O', 'Q'): 0.2, ('0', 'Q'): 0.2,
        ('D', '0'): 0.2, ('D', 'O'): 0.2, ('8', 'B'): 0.2,
        ('1', 'I'): 0.2, ('1', 'T'): 0.3, ('I', 'T'): 0.3,
        ('1', 'L'): 0.3, ('5', 'S'): 0.2, ('2', 'Z'): 0.2,
        ('A', '4'): 0.3, ('G', '6'): 0.3, ('U', 'V'): 0.3,
        ('P', 'R'): 0.3, ('E', 'F'): 0.3, ('E', 'B'): 0.4,
        ('M', 'N'): 0.4, ('K', 'X'): 0.4
    }

    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1): dp[i][0] = float(i)
    for j in range(m + 1): dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                koszt = 0.0
            else:
                para = (s1[i - 1], s2[j - 1])
                para_odw = (s2[j - 1], s1[i - 1])
                koszt = kary.get(para, kary.get(para_odw, 1.0))

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,  # usunięcie
                dp[i][j - 1] + 1.0,  # wstawienie
                dp[i - 1][j - 1] + koszt  # zamiana z wagą
            )

    return dp[n][m]


class LPRNetJsonBenchmark:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Benchmark LPRNet startuje na: {self.device}")
        self.detector = YOLO(YOLO_MODEL)
        self.lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(torch.load(LPR_MODEL, map_location=self.device))
        self.lprnet.eval()

    def decode(self, preds):
        preds = preds.cpu().detach().numpy()[0]
        preb_label = [np.argmax(preds[:, j], axis=0) for j in range(preds.shape[1])]
        res = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1: res.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1: pre_c = c
                continue
            res.append(c)
            pre_c = c
        return "".join([CHARS[i] for i in res])

    def run(self):
        json_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.json')]

        TP, FP, FN = 0, 0, 0
        exact_matches = 0
        total_cer = 0.0
        total_weighted_cer = 0.0  # <--- NOWA ZMIENNA
        total_yolo_time = 0.0
        total_ocr_time = 0.0
        processed_images = 0

        print(f"\n{'#' * 80}")
        print(f"{'PLIK':<15} | {'JSON GT':<12} | {'PREDYKCJA':<12} | {'S. CER':<8} | {'W. CER':<8} | STATUS")
        print(f"{'-' * 80}")

        for j_file in json_files:
            try:
                with open(os.path.join(TEST_IMAGES_DIR, j_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_name = data['imagePath']
                img_path = os.path.join(TEST_IMAGES_DIR, img_name)

                if not os.path.exists(img_path):
                    img_path = os.path.join(TEST_IMAGES_DIR, j_file.replace('.json', '.jpg'))

                img = cv2.imread(img_path)
                if img is None: continue
                processed_images += 1

                if not data.get('lps'): continue
                chars_list = data['lps'][0].get('characters', [])
                gt_text = "".join([str(c['char_id']) for c in chars_list])
                gt_text = "".join(re.findall(r'[A-Z0-9]', gt_text.upper()))

                if not gt_text: continue

                # 1. Detekcja YOLO
                t_yolo_start = time.perf_counter()
                results = self.detector(img, conf=0.3, verbose=False)[0]
                total_yolo_time += (time.perf_counter() - t_yolo_start)

                pred_text = "N/A"
                cer = 1.0
                w_cer = 1.0

                if len(results.boxes) > 0:
                    box = results.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]

                    # 2. OCR LPRNet
                    t_ocr_start = time.perf_counter()

                    im = cv2.resize(crop, (IMG_SIZE[0], IMG_SIZE[1])).astype('float32')
                    im = (im - 127.5) * 0.0078125
                    im = np.transpose(im, (2, 0, 1))
                    tensor = torch.from_numpy(im).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.lprnet(tensor)
                    pred_text = self.decode(logits)

                    total_ocr_time += (time.perf_counter() - t_ocr_start)

                    # --- KALKULACJA CER ORAZ WAŻONEGO CER ---
                    dlugosc_gt = len(gt_text) if len(gt_text) > 0 else 1.0

                    # Standardowy CER
                    edit_dist = Levenshtein.distance(gt_text, pred_text)
                    cer = edit_dist / dlugosc_gt
                    total_cer += cer

                    # Twoja nowa funkcja: Ważony CER
                    weighted_dist = wazony_levenshtein(gt_text, pred_text)
                    w_cer = weighted_dist / dlugosc_gt
                    total_weighted_cer += w_cer

                    if pred_text == gt_text:
                        TP += 1
                        exact_matches += 1
                        status = "✅ OK"
                    else:
                        FP += 1
                        status = "❌ BŁĄD"
                else:
                    FN += 1
                    status = "⚪ BRAK"

                print(f"{img_name:<15} | {gt_text:<12} | {pred_text:<12} | {cer:<8.2f} | {w_cer:<8.2f} | {status}")

            except Exception as e:
                print(f"Błąd przy pliku {j_file}: {e}")

        # --- WYPROWADZENIE METRYK KOŃCOWYCH ---
        if processed_images > 0:
            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

            acc = (exact_matches / processed_images) * 100

            # Średnie błędy znaków
            mianownik_ocr = (TP + FP) if (TP + FP) > 0 else 1.0
            avg_cer = total_cer / mianownik_ocr
            avg_weighted_cer = total_weighted_cer / mianownik_ocr  # <--- NOWA METRYKA

            yolo_ms = (total_yolo_time / processed_images) * 1000
            ocr_ms = (total_ocr_time / processed_images) * 1000
            e2e_ms = yolo_ms + ocr_ms

            print(f"{'-' * 80}")
            print(f"PODSUMOWANIE PURE LPRNet OCR:")
            print(f"Klasyfikacja OCR   -> F1: {F1:.4f} (R: {Recall:.2f}, P: {Precision:.2f})")
            print(f"Dokładność (Exact) -> {acc:.2f}%")
            print(f"Standardowy CER    -> {avg_cer:.4f}")
            print(f"WAŻONY CER (Optyk) -> {avg_weighted_cer:.4f} <--- (Taryfa ulgowa)")
            print(f"Czasy procesowe    -> YOLO: {yolo_ms:.1f}ms | LPRNet: {ocr_ms:.1f}ms")
            print(f"{'#' * 80}\n")

            # --- ZAPIS IDENTYCZNEGO RAPORTU DLA SHOW_RESULTS.PY ---
            os.makedirs(RESULTS_DIR, exist_ok=True)
            filename_res = f"results_lprnet_final_{int(time.time())}.txt"
            with open(os.path.join(RESULTS_DIR, filename_res), 'w', encoding='utf-8') as f:
                f.write(f"TP:{TP}\n")
                f.write(f"FP:{FP}\n")
                f.write(f"FN:{FN}\n")
                f.write(f"Precision:{Precision}\n")
                f.write(f"Recall:{Recall}\n")
                f.write(f"F1:{F1}\n")
                f.write(f"Plate_Accuracy:{acc}\n")
                f.write(f"CER:{avg_weighted_cer}\n")  # Zachowane jako główny CER dla starych skryptów
                f.write(f"Standard_CER:{avg_cer}\n")  # <--- WPIS DLA NOWEGO WYKRESU
                f.write(f"Weighted_CER:{avg_weighted_cer}\n")  # <--- WPIS DLA NOWEGO WYKRESU
                f.write(f"YOLO_ms:{yolo_ms}\n")
                f.write(f"OCR_ms:{ocr_ms}\n")
                f.write(f"E2E_ms:{e2e_ms}\n")

            print(f"[+] Zapisano zunifikowany raport z jawnymi metrykami CER: {RESULTS_DIR}/{filename_res}")


if __name__ == "__main__":
    LPRNetJsonBenchmark().run()