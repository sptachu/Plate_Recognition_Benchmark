import os
import time
import json
import cv2
import Levenshtein
import torch
import re
import hyperlpr3 as lpr3

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 100
IMAGES_DIR = 'dataset/UC3M-LP/train/'
LABELS_DIR = 'dataset/UC3M-LP/train/'


# --- FUNKCJE POMOCNICZE ---
def oblicz_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def is_partially_correct(true_txt, pred_txt):
    # Czyścimy oba teksty z masek
    t = true_txt.replace('?', '')
    p = pred_txt.replace('?', '')
    # Jeśli prawda zawiera się w predykcji (lub odwrotnie)
    return t in p or p in t


def clean_hyperlpr_output(text):
    """Usuwa chińskie znaki i zostawia tylko Alfanumeryczne + maskę."""
    # Usuwamy znaki chińskie (zakres Unicode)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    # Zostawiamy tylko A-Z, 0-9 i znak zapytania (maskę)
    return re.sub(r'[^A-Z0-9?]', '', text.upper())


def align_to_spanish_template(raw_text):
    """Próbuje dopasować surowy tekst do formatu 4 cyfry + 3 litery."""
    clean = clean_hyperlpr_output(raw_text)

    digits = "".join(re.findall(r'\d', clean))
    letters = "".join(re.findall(r'[A-Z]', clean))

    # Wybieramy pierwsze 4 cyfry i 3 litery (lub uzupełniamy maską)
    f_digits = digits[:4].ljust(4, '?')
    f_letters = letters[:3].ljust(3, '?')

    return f"{f_digits}{f_letters}"


def evaluate_masked_text(true_txt, pred_txt, mask_char='?'):
    """
    Ewaluacja odporna na przesunięcia (Shift-invariant).
    Ignoruje pozycje, które w Ground Truth są zamaskowane.
    """
    if not true_txt or not pred_txt:
        return False, 1.0

    # Tworzymy listę znaków, które faktycznie były widoczne w GT
    v_true = ""
    v_pred = ""

    # Najpierw normalizujemy długości do porównania (padding)
    max_len = max(len(true_txt), len(pred_txt))
    t_padded = true_txt.ljust(max_len, mask_char)
    p_padded = pred_txt.ljust(max_len, mask_char)

    for i in range(len(t_padded)):
        if t_padded[i] != mask_char:
            v_true += t_padded[i]
            v_pred += p_padded[i]

    # Masked Accuracy
    is_match = (v_true == v_pred)

    # Masked CER (Character Error Rate)
    dist = Levenshtein.distance(v_true, v_pred)
    cer = dist / len(v_true) if len(v_true) > 0 else 1.0

    return is_match, cer


def czytaj_ground_truth_json(json_path):
    if not os.path.exists(json_path): return [], []
    gt_boxes, gt_texts = [], []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for lp in data.get('lps', []):
        poly = lp.get('poly_coord', [])
        if len(poly) == 4:
            x_coords, y_coords = [p[0] for p in poly], [p[1] for p in poly]
            gt_boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])

        # UC3M-LP: characters często zawierają informację o maskowaniu
        chars = lp.get('characters', [])
        txt = "".join([c.get('char_id', '?') for c in chars])  # Zastąp puste/ukryte znaki przez ?
        gt_texts.append(txt.replace(" ", "").upper())
    return gt_boxes, gt_texts


# --- INICJALIZACJA ---
print("[*] Inicjalizacja HyperLPR3...")
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)

TP, FP, FN = 0, 0, 0
exact_matches, ocr_evaluated_count = 0, 0
total_cer, total_e2e_time, processed_images = 0.0, 0.0, 0

print(f"[*] Start ewaluacji: {IMAGES_DIR}\n")

try:
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        img_path = os.path.join(IMAGES_DIR, filename)
        json_path = os.path.join(LABELS_DIR, os.path.splitext(filename)[0] + '.json')

        img = cv2.imread(img_path)
        if img is None: continue
        gt_boxes, gt_texts = czytaj_ground_truth_json(json_path)
        if not gt_boxes: continue

        t_start = time.perf_counter()
        hy_results = catcher(img)
        total_e2e_time += (time.perf_counter() - t_start)

        matched_gt = set()
        for res in hy_results:
            raw_text, conf, p_type, p_box = res
            best_iou, best_gt_idx = 0.45, -1

            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt: continue
                iou = oblicz_iou(p_box, g_box)
                if iou >= best_iou:
                    best_iou, best_gt_idx = iou, g_idx

            if best_gt_idx != -1:
                TP += 1
                matched_gt.add(best_gt_idx)

                true_txt = gt_texts[best_gt_idx]
                # Używamy ulepszonego szablonu hiszpańskiego
                pred_txt = align_to_spanish_template(raw_text)

                is_match, m_cer = evaluate_masked_text(true_txt, pred_txt)

                total_cer += m_cer
                ocr_evaluated_count += 1
                if is_match: exact_matches += 1

                if is_partially_correct(true_txt, pred_txt):
                    print("⚠️ TRAFIENIE CZĘŚCIOWE")
                status = "✅" if is_match else "❌"
                print(f"{status} GT: '{true_txt}' | Pred: '{pred_txt}' | CER: {m_cer:.2f}")
            else:
                FP += 1

        FN += (len(gt_boxes) - len(matched_gt))
        processed_images += 1
        print(f"Postęp: {processed_images}/{MAX_IMAGES}", end='\r')
        if processed_images >= MAX_IMAGES: break

except Exception as e:
    print(f"\n[!] Błąd krytyczny: {e}")

finally:
    if processed_images > 0:
        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0.0
        avg_cer = total_cer / ocr_evaluated_count if ocr_evaluated_count > 0 else 1.0
        acc = (exact_matches / ocr_evaluated_count * 100) if ocr_evaluated_count > 0 else 0.0

        print("\n" + "=" * 50)
        print(f"OBRAZY: {processed_images} | DETEKCJA F1: {F1:.4f}")
        print(f"OCR ACC (MASKED): {acc:.2f}% | AVG CER: {avg_cer:.4f}")
        print(f"LATENCY E2E: {(total_e2e_time / processed_images) * 1000:.2f} ms")
        print("=" * 50)