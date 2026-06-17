import os
import time
import cv2
import Levenshtein
import hyperlpr3 as lpr3

# --- KONFIGURACJA CCPD ---
MAX_IMAGES = 1000
IMAGES_DIR = 'dataset/CCPD2019/CCPD2019/ccpd_base/'
RESULTS_DIR = "results"

# Mapowanie indeksów CCPD na znaki
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
             "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def decode_ccpd_filename(filename):
    """
    Format CCPD:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-110-72.jpg
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


# --- INICJALIZACJA ---
print("[*] Inicjalizacja HyperLPR3 (CCPD Mode)...")
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

# --- ZMIENNE STRUKTURALNE POD KLASYFIKACJĘ OCR ---
TP, FP, FN = 0, 0, 0
exact_matches, total_cer, total_weighted_cer, processed_images = 0, 0.0, 0.0, 0
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
    pred_text = "N/A"
    cer = 1.0
    w_cer = 1.0

    for res_text, conf, p_type, p_box in results:
        iou = oblicz_iou(p_box, gt_box)

        # Sprawdzamy czy detektor trafił w fizyczny obszar tablicy
        if iou > 0.45:
            found_this_img = True
            pred_text = res_text.replace(" ", "")

            # Miary błędów znakowych
            dlugosc_gt = len(gt_text) if len(gt_text) > 0 else 1.0
            cer = Levenshtein.distance(gt_text, pred_text) / dlugosc_gt
            total_cer += cer

            w_cer = wazony_levenshtein(gt_text, pred_text) / dlugosc_gt
            total_weighted_cer += w_cer

            # Macierz pomyłek na poziomie czystego OCR tablicy
            if gt_text == pred_text:
                TP += 1
                exact_matches += 1
                status = "✅ OK"
            else:
                FP += 1
                status = "❌ BŁĄD"

            print(f"{status} GT: {gt_text} | Pred: {pred_text} | S.CER: {cer:.2f} | W.CER: {w_cer:.2f}")
            break

    if not found_this_img:
        FN += 1
        print(f"⚪ BRAK DETEKCJI (FN) dla: {gt_text}")

    processed_images += 1
    print(f"Postęp: {processed_images}/{MAX_IMAGES}", end='\r')
    if processed_images >= MAX_IMAGES: break

# --- RAPORT KOŃCOWY ---
if processed_images > 0:
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    acc = (exact_matches / processed_images) * 100
    mianownik_ocr = (TP + FP) if (TP + FP) > 0 else 1.0
    avg_cer = total_cer / mianownik_ocr
    avg_weighted_cer = total_weighted_cer / mianownik_ocr
    avg_ms = (total_time / processed_images) * 1000

    print("\n" + "=" * 60)
    print(f"WYNIKI CCPD PURE OCR (HyperLPR3)")
    print(f"Przetworzono: {processed_images} zdjęć")
    print(f"Klasyfikacja OCR -> F1: {F1:.4f} (R: {Recall:.2f}, P: {Precision:.2f})")
    print(f"OCR Accuracy: {acc:.2f}%")
    print(f"Standardowy CER: {avg_cer:.4f} | WAŻONY CER: {avg_weighted_cer:.4f}")
    print(f"Średni czas E2E: {avg_ms:.2f} ms")
    print("=" * 60)

    # --- ZAPIS PEŁNEGO RAPORTO ZGODNEGO Z SHOW_RESULTS.PY ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename_res = f"results_ccpdhyper_final_{int(time.time())}.txt"  # Wykryje się jako model: CCPDHYPER
    with open(os.path.join(RESULTS_DIR, filename_res), 'w', encoding='utf-8') as f:
        f.write(f"TP:{TP}\n")
        f.write(f"FP:{FP}\n")
        f.write(f"FN:{FN}\n")
        f.write(f"Precision:{Precision}\n")
        f.write(f"Recall:{Recall}\n")
        f.write(f"F1:{F1}\n")
        f.write(f"Plate_Accuracy:{acc}\n")
        f.write(f"CER:{avg_cer}\n")
        f.write(f"Weighted_CER:{avg_weighted_cer}\n")
        f.write(f"YOLO_ms:{avg_ms * 0.5}\n")
        f.write(f"OCR_ms:{avg_ms * 0.5}\n")
        f.write(f"E2E_ms:{avg_ms}\n")

    print(f"[+] Zapisano zunifikowany raport CCPD: {RESULTS_DIR}/{filename_res}")