import os
import time
import cv2
import torch
import numpy as np
import Levenshtein
from ultralytics import YOLO
from model_lprnet.LPRNet import build_lprnet

# --- KONFIGURACJA CCPD ---
YOLO_MODEL = 'yolo11_plate.pt'
LPR_MODEL = './weights_lprnet/Final_LPRNet_model.pth'
IMAGES_DIR = './dataset/CCPD2019/CCPD2019/ccpd_base/'
RESULTS_DIR = "results"
MAX_IMAGES = 1000
IMG_SIZE = [94, 24]

# --- SŁOWNIKI ZNAKÓW CCPD (DEDYKOWANE DO DEKODOWANIA NAZW PLIKÓW Z DYSKU) ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- JEDYNY, ZUNIFIKOWANY SŁOWNIK MODELU (68 KLAS DO METODY DECODE) ---
CHARS_CCPD = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "-", "_"
]


def decode_ccpd_filename(filename):
    """
    Bezpieczne i zgodne z oficjalną dokumentacją dekodowanie metryk CCPD.
    """
    try:
        name = os.path.splitext(filename)[0]
        parts = name.split('-')

        if len(parts) < 5:
            return None, None

        # 1. Dekodowanie Bounding Boxa [x1, y1, x2, y2]
        box_coords = parts[2].split('_')
        x1y1 = box_coords[0].split('&')
        x2y2 = box_coords[1].split('&')
        gt_box = [int(x1y1[0]), int(x1y1[1]), int(x2y2[0]), int(x2y2[1])]

        # 2. Inteligentne szukanie sekcji z indeksami znaków
        sekcja_tekstu = None
        for part in parts:
            if len(part.split('_')) == 7:
                sekcja_tekstu = part
                break

        if sekcja_tekstu is None:
            sekcja_tekstu = parts[4]

        indices = [int(x) for x in sekcja_tekstu.split('_')]

        # --- DEKODOWANIE ZGODNE ZE STRUKTURĄ TRZECH SŁOWNIKÓW ---
        if indices[0] >= len(provinces): return None, None
        res_text = provinces[indices[0]]

        if indices[1] >= len(alphabets): return None, None
        res_text += alphabets[indices[1]]

        for idx in indices[2:]:
            if idx >= len(ads): return None, None
            res_text += ads[idx]

        return gt_box, res_text
    except Exception:
        return None, None


def oblicz_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def wazony_levenshtein(s1, s2):
    """Profesjonalne wagi dla systemów OCR na tablicach rejestracyjnych."""
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
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + koszt)
    return dp[n][m]


class LPRNetCCPDPruner:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[*] Benchmark LPRNet (ORIGINAL CCPD) startuje na: {self.device}")
        self.detector = YOLO(YOLO_MODEL)

        self.lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(torch.load(LPR_MODEL, map_location=self.device))
        self.lprnet.eval()

    def decode(self, preds):
        preds = preds.cpu().detach().numpy()[0]
        preb_label = [np.argmax(preds[:, j], axis=0) for j in range(preds.shape[1])]
        res = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS_CCPD) - 1: res.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS_CCPD) - 1):
                if c == len(CHARS_CCPD) - 1: pre_c = c
                continue
            res.append(c)
            pre_c = c
        return "".join([CHARS_CCPD[i] for i in res])

    def run(self):
        if not os.path.exists(IMAGES_DIR):
            print(f"[!] Ścieżka {IMAGES_DIR} nie istnieje!")
            return

        files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        TP, FP, FN = 0, 0, 0
        exact_matches = 0
        total_cer = 0.0
        total_weighted_cer = 0.0
        total_yolo_time = 0.0
        total_ocr_time = 0.0
        processed_images = 0

        print(f"\n{'#' * 85}")
        print(f"{'GT TEXT':<12} | {'PREDYKCJA':<12} | {'S. CER':<6} | {'W. CER':<6} | {'IoU':<5} | STATUS")
        print(f"{'-' * 85}")

        for filename in files:
            try:
                gt_box, gt_text = decode_ccpd_filename(filename)
                if gt_box is None or gt_text is None: continue

                img_path = os.path.join(IMAGES_DIR, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                processed_images += 1

                # 1. Detekcja YOLO
                t_yolo_start = time.perf_counter()
                results = self.detector(img, conf=0.3, verbose=False)[0]
                total_yolo_time += (time.perf_counter() - t_yolo_start)

                pred_text = "N/A"
                cer = 1.0
                w_cer = 1.0
                max_iou = 0.0
                found_plate_area = False

                for box in results.boxes:
                    p_box = list(map(int, box.xyxy[0]))
                    iou = oblicz_iou(p_box, gt_box)
                    if iou > max_iou: max_iou = iou

                    if iou > 0.45:
                        found_plate_area = True
                        x1, y1, x2, y2 = p_box
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
                        break

                dlugosc_gt = len(gt_text) if len(gt_text) > 0 else 1.0

                if found_plate_area:
                    edit_dist = Levenshtein.distance(gt_text, pred_text)
                    cer = edit_dist / dlugosc_gt
                    total_cer += cer

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
                    status = "⚪ BRAK DETEKCJI"

                print(f"{gt_text:<12} | {pred_text:<12} | {cer:<6.2f} | {w_cer:<6.2f} | {max_iou:<5.2f} | {status}")

                if processed_images >= MAX_IMAGES: break

            except Exception as e:
                print(f"Błąd przy pliku {filename}: {e}")

        # --- WYPROWADZENIE METRYK KOŃCOWYCH ---
        if processed_images > 0:
            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

            acc = (exact_matches / processed_images) * 100
            mianownik_ocr = (TP + FP) if (TP + FP) > 0 else 1.0
            avg_cer = total_cer / mianownik_ocr
            avg_weighted_cer = total_weighted_cer / mianownik_ocr

            yolo_ms = (total_yolo_time / processed_images) * 1000
            ocr_ms = (total_ocr_time / processed_images) * 1000
            e2e_ms = yolo_ms + ocr_ms

            print(f"{'-' * 85}")
            print(f"PODSUMOWANIE ORYGINALNEGO LPRNET (CCPD):")
            print(f"Klasyfikacja OCR -> F1: {F1:.4f} (R: {Recall:.2f}, P: {Precision:.2f})")
            print(f"Dokładność (Exact Match): {acc:.2f}%")
            print(f"Standardowy CER: {avg_cer:.4f} | WAŻONY CER: {avg_weighted_cer:.4f}")
            print(f"Latency -> YOLO: {yolo_ms:.1f}ms | LPRNet: {ocr_ms:.1f}ms | E2E: {e2e_ms:.1f}ms")
            print(f"{'#' * 85}\n")

            os.makedirs(RESULTS_DIR, exist_ok=True)
            filename_res = f"results_origccpd_final_{int(time.time())}.txt"
            with open(os.path.join(RESULTS_DIR, filename_res), 'w', encoding='utf-8') as f:
                f.write(f"TP:{TP}\n")
                f.write(f"FP:{FP}\n")
                f.write(f"FN:{FN}\n")
                f.write(f"Precision:{Precision}\n")
                f.write(f"Recall:{Recall}\n")
                f.write(f"F1:{F1}\n")
                f.write(f"Plate_Accuracy:{acc}\n")
                f.write(f"CER:{avg_weighted_cer}\n")
                f.write(f"Standard_CER:{avg_cer}\n")
                f.write(f"Weighted_CER:{avg_weighted_cer}\n")
                f.write(f"YOLO_ms:{yolo_ms}\n")
                f.write(f"OCR_ms:{ocr_ms}\n")
                f.write(f"E2E_ms:{e2e_ms}\n")

            print(f"[+] Zapisano zunifikowany raport: {RESULTS_DIR}/{filename_res}")


if __name__ == "__main__":
    LPRNetCCPDPruner().run()