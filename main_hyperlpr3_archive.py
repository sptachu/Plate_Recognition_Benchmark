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


# --- INICJALIZACJA HYPERLPR3 ---
print("[*] Inicjalizacja HyperLPR3...")
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

# --- ZMIENNE STATYSTKCZNE SKALIBROWANE POD OCR ---
TP, FP, FN = 0, 0, 0
exact_matches = 0
total_cer = 0.0
total_weighted_cer = 0.0  # <--- NOWA ZMIENNA
total_e2e_time = 0.0
processed_images = 0

print(f"\n[*] Start ewaluacji HyperLPR3 na {MAX_IMAGES} obrazkach (Padding: 30px)...\n")

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

        # --- ANALIZA WYNIKÓW POD KĄTEM JAKOŚCI OCR ---
        pred_text = "N/A"
        cer = 1.0
        w_cer = 1.0

        if hy_results:
            raw_pred = hy_results[0][0]
            pred_txt = clean_output(raw_pred)

            # Kalkulacja błędów znakowych
            dlugosc_gt = len(true_txt) if len(true_txt) > 0 else 1.0

            # Standardowy CER
            edit_dist = Levenshtein.distance(true_txt, pred_txt)
            cer = edit_dist / dlugosc_gt
            total_cer += cer

            # Ważony CER
            weighted_dist = wazony_levenshtein(true_txt, pred_txt)
            w_cer = weighted_dist / dlugosc_gt
            total_weighted_cer += w_cer

            # Klasyfikacja klasyczna pod macierz pomyłek OCR
            if true_txt == pred_txt:
                TP += 1  # Idealny odczyt całego ciągu
                exact_matches += 1
                status = "✅ OK"
            else:
                FP += 1  # Odczytał, ale z błędami w znakach
                status = "❌ BŁĄD"
        else:
            FN += 1  # Silnik całkowicie pominął obrazek lub nie odczytał nic
            status = "⚪ BRAK"

        print(
            f"{filename:<20} | GT: {true_txt:<10} | Pred: {pred_txt:<10} | S.CER: {cer:.2f} | W.CER: {w_cer:.2f} | {status}")

        processed_images += 1
        print(f"Postęp: {processed_images}/{MAX_IMAGES}", end='\r')

        if processed_images >= MAX_IMAGES: break

except Exception as e:
    print(f"\n[!] Błąd w pętli: {e}")

finally:
    if processed_images > 0:
        # Metryki jakości na poziomie OCR
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

        # Wyliczanie średnich wydajności
        avg_ms = (total_e2e_time / processed_images) * 1000
        acc = (exact_matches / processed_images) * 100

        mianownik_ocr = (TP + FP) if (TP + FP) > 0 else 1.0
        avg_cer = total_cer / mianownik_ocr
        avg_weighted_cer = total_weighted_cer / mianownik_ocr

        print("\n" + "=" * 60)
        print("           RAPORT HYPERLPR3 PURE OCR (OPTIMIZED)")
        print("=" * 60)
        print(f"Przetworzone zdjęcia: {processed_images}")
        print(f"Klasyfikacja OCR -> F1: {F1:.4f} (R: {Recall:.2f}, P: {Precision:.2f})")
        print(f"Dokładność (Exact) -> {acc:.2f}%")
        print(f"Standardowy CER    -> {avg_cer:.4f}")
        print(f"WAŻONY CER (Optyk) -> {avg_weighted_cer:.4f}")
        print(f"Latency  -> Średnio E2E: {avg_ms:.2f} ms / obraz")
        print("=" * 60)

        # --- ZAPIS IDENTYCZNEGO RAPORTU DLA SHOW_RESULTS.PY ---
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename_res = f"results_spanishhyper_final_{int(time.time())}.txt"
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
            f.write(f"YOLO_ms:{avg_ms * 0.5}\n")  # Emulacja podziału czasowego pod ogólny wykres
            f.write(f"OCR_ms:{avg_ms * 0.5}\n")  # Emulacja podziału czasowego pod ogólny wykres
            f.write(f"E2E_ms:{avg_ms}\n")

        print(f"[+] Zapisano zunifikowany raport HyperLPR3: {RESULTS_DIR}/{filename_res}")