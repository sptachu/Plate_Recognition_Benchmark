# import os
# import numpy as np
# import matplotlib
# # Naprawa błędu PyCharma: Wymuszenie użycia silnika okienkowego Tkinter
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# RESULTS_DIR = "results"
#
#
# def znajdz_najnowszy_raport():
#     if not os.path.exists(RESULTS_DIR):
#         print(f"Folder '{RESULTS_DIR}' nie istnieje. Uruchom najpierw main.py!")
#         return None
#
#     pliki = [f for f in os.listdir(RESULTS_DIR) if f.startswith("results_hyperlpr3_") and f.endswith(".txt")]
#     if not pliki:
#         print(f"Brak plików z wynikami w folderze '{RESULTS_DIR}'.")
#         return None
#
#     def wyciagnij_nr(nazwa):
#         try:
#             return int(nazwa.split('_')[-1].split('.')[0])
#         except ValueError:
#             return -1
#
#     najnowszy = max(pliki, key=wyciagnij_nr)
#     return os.path.join(RESULTS_DIR, najnowszy)
#
#
# def wczytaj_dane(sciezka):
#     dane = {}
#     with open(sciezka, 'r', encoding='utf-8') as f:
#         for linia in f:
#             if ':' in linia:
#                 klucz, wartosc = linia.strip().split(':', 1)
#                 dane[klucz] = float(wartosc)
#     return dane
#
#
# def rysuj_wykresy():
#     sciezka = znajdz_najnowszy_raport()
#     if not sciezka:
#         return
#
#     print(f"Wczytywanie raportu: {os.path.basename(sciezka)}...")
#     dane = wczytaj_dane(sciezka)
#
#     # Ustawienie globalnego stylu z Seaborn
#     sns.set_theme(style="whitegrid")
#     fig = plt.figure(figsize=(16, 6))
#     fig.canvas.manager.set_window_title(f"Analiza ALPR - {os.path.basename(sciezka)}")
#
#     # --- 1. WYKRES: Wydajność Czasowa (Latency) ---
#     ax1 = plt.subplot(1, 3, 1)
#     czasy = [dane['YOLO_ms'], dane['OCR_ms'], dane['E2E_ms']]
#     etykiety_czas = ['Detekcja YOLO', 'Odczyt OCR', 'Całość E2E']
#
#     # Naprawa warninga z seaborn (dodano hue i legend=False)
#     bars1 = sns.barplot(x=etykiety_czas, y=czasy, hue=etykiety_czas, palette="rocket", legend=False, ax=ax1)
#     ax1.set_title("Wydajność (Czas Przetwarzania)", fontsize=14, pad=15)
#     ax1.set_ylabel("Czas [milisekundy]")
#
#     for bar in bars1.containers[0]:
#         ax1.annotate(f"{bar.get_height():.1f} ms",
#                      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
#                      xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
#
#     # --- 2. WYKRES: Metryki Jakości ---
#     ax2 = plt.subplot(1, 3, 2)
#     # Przeliczamy P, R, F1 na procenty (0-100), żeby ładnie wyglądały obok Plate_Accuracy
#     metryki_wartosci = [dane['Precision'] * 100, dane['Recall'] * 100, dane['F1'] * 100, dane['Plate_Accuracy']]
#     etykiety_metryk = ['Precyzja', 'Czułość', 'F1-Score', 'Exact Match\n(OCR)']
#
#     # Naprawa warninga z seaborn (dodano hue i legend=False)
#     bars2 = sns.barplot(x=etykiety_metryk, y=metryki_wartosci, hue=etykiety_metryk, palette="viridis", legend=False, ax=ax2)
#     ax2.set_title("Jakość Modeli (%)", fontsize=14, pad=15)
#     ax2.set_ylim(0, 110)  # Skala wymuszona do 100%
#     ax2.set_ylabel("Skuteczność [%]")
#
#     for bar in bars2.containers[0]:
#         ax2.annotate(f"{bar.get_height():.1f}%",
#                      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
#                      xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
#
#     # --- 3. WYKRES: Macierz Pomyłek Detekcji (Confusion Matrix) ---
#     ax3 = plt.subplot(1, 3, 3)
#     # W detekcji obiektów nie mierzymy "True Negative" (Tło rozpoznane jako tło).
#     # Macierz to: [[TP, FN], [FP, Puste]]
#     cm = np.array([
#         [dane['TP'], dane['FN']],
#         [dane['FP'], np.nan]  # N/A dla TN
#     ])
#
#     sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", cbar=False, ax=ax3,
#                 annot_kws={"size": 16, "weight": "bold"},
#                 mask=np.isnan(cm))  # Ukrywamy komórkę True Negative
#
#     ax3.set_title("Macierz Pomyłek (Detekcja)", fontsize=14, pad=15)
#     ax3.set_xlabel("Przewidywanie modelu", fontsize=12)
#     ax3.set_ylabel("Rzeczywistość (Ground Truth)", fontsize=12)
#     ax3.set_xticklabels(['Tablica', 'Tło'])
#     ax3.set_yticklabels(['Tablica', 'Tło'])
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     rysuj_wykresy()

import os
import cv2
import numpy as np
import matplotlib

# Naprawa błędu PyCharma: Wymuszenie użycia silnika okienkowego Tkinter
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"


def znajdz_najnowszy_raport():
    if not os.path.exists(RESULTS_DIR):
        print(f"Folder '{RESULTS_DIR}' nie istnieje. Uruchom najpierw main.py!")
        return None

    # Obsługuje zarówno pliki hyperlpr3, jak i inne tekstówki z wynikami
    pliki = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".txt")]
    if not pliki:
        print(f"Brak plików z wynikami w folderze '{RESULTS_DIR}'.")
        return None

    def wyciagnij_nr(nazwa):
        # Wyciąganie timestampu na końcu nazwy pliku
        liczby = re.findall(r'\d+', nazwa)
        return int(liczby[-1]) if liczby else 0

    import re
    najnowszy = max(pliki, key=wyciagnij_nr)
    return os.path.join(RESULTS_DIR, najnowszy)


def wczytaj_dane(sciezka):
    dane = {}
    with open(sciezka, 'r', encoding='utf-8') as f:
        for linia in f:
            if ':' in linia:
                klucz, wartosc = linia.strip().split(':', 1)
                try:
                    dane[klucz.strip()] = float(wartosc.strip())
                except ValueError:
                    continue
    return dane


def rysuj_wykresy():
    sciezka = znajdz_najnowszy_raport()
    if not sciezka:
        return

    print(f"Wczytywanie raportu: {os.path.basename(sciezka)}...")
    dane = wczytaj_dane(sciezka)

    # Ustawienie globalnego stylu z Seaborn
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 6))
    fig.canvas.manager.set_window_title(f"Analiza ALPR - {os.path.basename(sciezka)}")

    # --- 1. WYKRES: Wydajność Czasowa (Zabezpieczony przed KeyError) ---
    ax1 = plt.subplot(1, 3, 1)

    # Bezpieczne wyciąganie ze słownika metodą .get(klucz, domyslna_wartosc)
    yolo_time = dane.get('YOLO_ms', 0.0)
    ocr_time = dane.get('OCR_ms', 0.0)
    e2e_time = dane.get('E2E_ms', 0.0)

    # Mały trik: jeśli nie ma cząstkowych czasów, ale jest E2E, dajemy znać na wykresie
    if e2e_time > 0 and yolo_time == 0.0 and ocr_time == 0.0:
        yolo_time = e2e_time * 0.4  # Szacunkowe 40% czasu na detekcję
        ocr_time = e2e_time * 0.6  # Szacunkowe 60% czasu na OCR
        ax1.set_xlabel("*Czasy YOLO i OCR oszacowane proporcjonalnie", color="orange")

    czasy = [yolo_time, ocr_time, e2e_time]
    etykiety_czas = ['Detekcja YOLO', 'Odczyt OCR', 'Całość E2E']

    bars1 = sns.barplot(x=etykiety_czas, y=czasy, hue=etykiety_czas, palette="rocket", legend=False, ax=ax1)
    ax1.set_title("Wydajność (Czas Przetwarzania)", fontsize=14, pad=15)
    ax1.set_ylabel("Czas [milisekundy]")

    for bar in bars1.containers[0]:
        ax1.annotate(f"{bar.get_height():.1f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # --- 2. WYKRES: Metryki Jakości ---
    ax2 = plt.subplot(1, 3, 2)
    # Zabezpieczenie parametrów jakościowych
    p_val = dane.get('Precision', 0.0) * 100
    r_val = dane.get('Recall', 0.0) * 100
    f1_val = dane.get('F1', 0.0) * 100
    acc_val = dane.get('Plate_Accuracy', 0.0)

    metryki_wartosci = [p_val, r_val, f1_val, acc_val]
    etykiety_metryk = ['Precyzja', 'Czułość', 'F1-Score', 'Exact Match\n(OCR)']

    bars2 = sns.barplot(x=etykiety_metryk, y=metryki_wartosci, hue=etykiety_metryk, palette="viridis", legend=False,
                        ax=ax2)
    ax2.set_title("Jakość Modeli (%)", fontsize=14, pad=15)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Skuteczność [%]")

    for bar in bars2.containers[0]:
        ax2.annotate(f"{bar.get_height():.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # --- 3. WYKRES: Macierz Pomyłek Detekcji ---
    ax3 = plt.subplot(1, 3, 3)
    cm = np.array([
        [dane.get('TP', 0), dane.get('FN', 0)],
        [dane.get('FP', 0), np.nan]
    ])

    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", cbar=False, ax=ax3,
                annot_kws={"size": 16, "weight": "bold"},
                mask=np.isnan(cm))

    ax3.set_title("Macierz Pomyłek (Detekcja)", fontsize=14, pad=15)
    ax3.set_xlabel("Przewidywanie modelu", fontsize=12)
    ax3.set_ylabel("Rzeczywistość (Ground Truth)", fontsize=12)
    ax3.set_xticklabels(['Tablica', 'Tło'])
    ax3.set_yticklabels(['Tablica', 'Tło'])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rysuj_wykresy()