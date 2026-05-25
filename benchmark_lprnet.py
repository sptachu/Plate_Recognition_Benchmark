import os
import time
import cv2
import torch
import numpy as np
import json
import re
import Levenshtein
from ultralytics import YOLO
from model_lprnet.LPRNet import build_lprnet
from load_data import CHARS

# --- KONFIGURACJA ---
YOLO_MODEL = 'yolo11_plate.pt'
LPR_MODEL = './weights/europe/LPRNet_Euro_Epoch_100.pth'
TEST_IMAGES_DIR = './data/UC3M-LP/test/'  # Folder ze zdjęciami i JSONami
RESULTS_DIR = "results"
IMG_SIZE = [94, 24]


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

        # --- SKALIBROWANE METRYKI POD LPRNET ---
        TP, FP, FN = 0, 0, 0
        exact_matches = 0
        total_cer = 0.0
        total_yolo_time = 0.0
        total_ocr_time = 0.0
        processed_images = 0

        print(f"\n{'#' * 70}")
        print(f"{'PLIK':<15} | {'JSON GT':<12} | {'PREDYKCJA':<12} | {'STATUS'}")
        print(f"{'-' * 70}")

        for j_file in json_files:
            try:
                # 1. Odczyt JSON
                with open(os.path.join(TEST_IMAGES_DIR, j_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_name = data['imagePath']
                img_path = os.path.join(TEST_IMAGES_DIR, img_name)

                if not os.path.exists(img_path):
                    img_path = os.path.join(TEST_IMAGES_DIR, j_file.replace('.json', '.jpg'))

                img = cv2.imread(img_path)
                if img is None: continue
                processed_images += 1

                # Wyciąganie Ground Truth z listy characters
                if not data.get('lps'): continue
                chars_list = data['lps'][0].get('characters', [])
                gt_text = "".join([str(c['char_id']) for c in chars_list])
                gt_text = "".join(re.findall(r'[A-Z0-9]', gt_text.upper()))

                if not gt_text: continue

                # 2. Detekcja YOLO
                t_yolo_start = time.perf_counter()
                results = self.detector(img, conf=0.3, verbose=False)[0]
                total_yolo_time += (time.perf_counter() - t_yolo_start)

                pred_text = "N/A"

                if len(results.boxes) > 0:
                    box = results.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]

                    # 3. OCR LPRNet
                    t_ocr_start = time.perf_counter()

                    im = cv2.resize(crop, (IMG_SIZE[0], IMG_SIZE[1])).astype('float32')
                    im = (im - 127.5) * 0.0078125
                    im = np.transpose(im, (2, 0, 1))
                    tensor = torch.from_numpy(im).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.lprnet(tensor)
                    pred_text = self.decode(logits)

                    total_ocr_time += (time.perf_counter() - t_ocr_start)

                    # Obliczanie błędu znaków CER
                    edit_dist = Levenshtein.distance(gt_text, pred_text)
                    total_cer += edit_dist / len(gt_text) if len(gt_text) > 0 else 1.0

                    # --- KLASYFIKACJA METRYK OCR (LPRNET) ---
                    if pred_text == gt_text:
                        TP += 1  # LPRNet trafił w dziesiątkę!
                        exact_matches += 1
                        status = "✅ OK"
                    else:
                        FP += 1  # LPRNet odczytał tablicę, ale popełnił błąd (False Positive)
                        status = "❌ BŁĄD"
                else:
                    FN += 1  # System w ogóle nie zwrócił wyniku dla tej tablicy (False Negative)
                    status = "⚪ BRAK"

                print(f"{img_name:<15} | {gt_text:<12} | {pred_text:<12} | {status}")

            except Exception as e:
                print(f"Błąd przy pliku {j_file}: {e}")

        # --- KALKULACJA METRYK KOŃCOWYCH DLA LPRNET ---
        if processed_images > 0:
            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

            acc = (exact_matches / processed_images) * 100
            # Średni CER liczymy tylko dla odczytanych ramek
            avg_cer = (total_cer / (TP + FP)) if (TP + FP) > 0 else 1.0

            yolo_ms = (total_yolo_time / processed_images) * 1000
            ocr_ms = (total_ocr_time / processed_images) * 1000
            e2e_ms = yolo_ms + ocr_ms

            print(f"{'-' * 70}")
            print(f"PODSUMOWANIE PURE LPRNet OCR:")
            print(f"Klasyfikacja OCR -> F1: {F1:.4f} (Czułość R: {Recall:.2f}, Precyzja P: {Precision:.2f})")
            print(f"Dokładność (Exact Match): {acc:.2f}% | Średni błąd znaków CER: {avg_cer:.4f}")
            print(f"Czasy procesowe -> YOLO: {yolo_ms:.1f}ms | LPRNet: {ocr_ms:.1f}ms")
            print(f"{'#' * 70}\n")

            # --- ZAPIS LOGU POD WYKRESY (SHOW_RESULTS.PY) ---
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
                f.write(f"CER:{avg_cer}\n")
                f.write(f"YOLO_ms:{yolo_ms}\n")
                f.write(f"OCR_ms:{ocr_ms}\n")
                f.write(f"E2E_ms:{e2e_ms}\n")

            print(f"[+] Zapisano czysty raport LPRNet: {RESULTS_DIR}/{filename_res}")


if __name__ == "__main__":
    LPRNetJsonBenchmark().run()