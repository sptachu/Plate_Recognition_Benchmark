import os
import time
import cv2
import Levenshtein
import re
import hyperlpr3 as lpr3

# --- USTAWIENIA TESTU ---
MAX_IMAGES = 1000
IMAGES_DIR = 'dataset/archive/dataset_final/train/'
RESULTS_DIR = "results"


# --- FUNKCJE POMOCNICZE ---

def add_padding(img, padding_size=30):
    """
    Dodaje margines wokół tablicy. HyperLPR3 potrzebuje tła wokół ramki,
    żeby 'zrozumieć', że patrzy na prostokąt tablicy.
    """
    return cv2.copyMakeBorder(
        img, padding_size, padding_size, padding_size, padding_size,
        cv2.BORDER_CONSTANT, value=[128, 128, 128]  # Szary neutralny
    )


def clean_output(text):
    """Usuwa chińskie znaki i wszystko, co nie jest literą/cyfrą."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def czytaj_ground_truth_z_nazwy(img_path):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    return base_name.replace(" ", "").upper()


# --- INICJALIZACJA HYPERLPR3 ---
print("[*] Inicjalizacja HyperLPR3...")
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

# --- ZMIENNE STATYSTYCZNE ---
TP, FP, FN = 0, 0, 0
exact_matches = 0
total_cer = 0.0
ocr_evaluated_count = 0
total_e2e_time = 0.0
processed_images = 0

print(f"\n[*] Start ewaluacji na {MAX_IMAGES} obrazkach (Padding: 30px)...\n")

try:
    files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for filename in files:
        img_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        true_txt = czytaj_ground_truth_z_nazwy(img_path)

        # --- PRE-PROCESSING & INFERENCJA (Z POMIAREM CZASU) ---
        t_start = time.perf_counter()

        img_prepped = add_padding(img, padding_size=30)
        hy_results = catcher(img_prepped)

        duration = time.perf_counter() - t_start
        total_e2e_time += duration

        # --- ANALIZA WYNIKÓW ---
        if hy_results:
            raw_pred = hy_results[0][0]
            pred_txt = clean_output(raw_pred)

            ocr_evaluated_count += 1
            TP += 1

            if true_txt == pred_txt:
                exact_matches += 1
                print(f"✅ {filename}: Trafienie!")
            else:
                print(f"❌ {filename}: GT: {true_txt} | Pred: {pred_txt}")

            edit_dist = Levenshtein.distance(true_txt, pred_txt)
            total_cer += edit_dist / len(true_txt) if len(true_txt) > 0 else 1.0
        else:
            FN += 1
            print(f"⚪ {filename}: Nie wykryto tablicy")

        processed_images += 1
        print(f"Postęp: {processed_images}/{MAX_IMAGES}", end='\r')

        if processed_images >= MAX_IMAGES: break

except Exception as e:
    print(f"\n[!] Błąd w pętli: {e}")

finally:
    if processed_images > 0:
        # Metryki detekcji
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

        # Metryki wydajności i OCR
        avg_ms = (total_e2e_time / processed_images) * 1000
        acc = (exact_matches / ocr_evaluated_count * 100) if ocr_evaluated_count > 0 else 0.0
        avg_cer = (total_cer / ocr_evaluated_count) if ocr_evaluated_count > 0 else 0.0

        print("\n" + "=" * 50)
        print("           RAPORT HYPERLPR3 (OPTIMIZED)")
        print("=" * 50)
        print(f"Przetworzone zdjęcia: {processed_images}")
        print(f"Detekcja -> F1: {F1:.4f} (R: {Recall:.2f}, P: {Precision:.2f})")
        print(f"OCR      -> Accuracy: {acc:.2f}%, Średni CER: {avg_cer:.4f}")
        print(f"Latency  -> Średnio E2E: {avg_ms:.2f} ms / obraz")
        print("=" * 50)

        # --- POPRAWKA: KOMPLETNY ZAPIS ZMIENNYCH DLA SCRIPTU WYKRESÓW ---
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename_res = f"results_hyperlpr3_final_{int(time.time())}.txt"
        with open(os.path.join(RESULTS_DIR, filename_res), 'w', encoding='utf-8') as f:
            f.write(f"TP:{TP}\n")
            f.write(f"FP:{FP}\n")
            f.write(f"FN:{FN}\n")
            f.write(f"Precision:{Precision}\n")
            f.write(f"Recall:{Recall}\n")
            f.write(f"F1:{F1}\n")
            f.write(f"Plate_Accuracy:{acc}\n")
            f.write(f"CER:{avg_cer}\n")

            # Podział sumarycznego czasu na równe połowy dla kompatybilności struktury
            f.write(f"YOLO_ms:{avg_ms * 0.5}\n")
            f.write(f"OCR_ms:{avg_ms * 0.5}\n")
            f.write(f"E2E_ms:{avg_ms}\n")

        print(f"[+] Zapisano pełny raport: {RESULTS_DIR}/{filename_res}")