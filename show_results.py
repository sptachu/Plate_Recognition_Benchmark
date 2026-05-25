import os
import cv2
import numpy as np
import re
import matplotlib

# Wymuszenie użycia silnika okienkowego Tkinter dla stabilności w PyCharm/VSCode
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"


def znajdz_najnowszy_raport():
    if not os.path.exists(RESULTS_DIR):
        print(f"Folder '{RESULTS_DIR}' nie istnieje. Uruchom najpierw main.py!")
        return None

    pliki = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".txt")]
    if not pliki:
        print(f"Brak plików z wynikami w folderze '{RESULTS_DIR}'.")
        return None

    def wyciagnij_nr(nazwa):
        # Wyciąganie timestampu na końcu nazwy pliku
        liczby = re.findall(r'\d+', nazwa)
        return int(liczby[-1]) if liczby else 0

    najnowszy = max(pliki, key=wyciagnij_nr)
    return os.path.join(RESULTS_DIR, najnowszy)


def wyciagnij_nazwe_modelu(sciezka_pliku):
    """
    Automatycznie wyciąga nazwę modelu z pliku tekstowego.
    Przykład: 'results_hyperlpr3_final_123.txt' -> 'HYPERLPR3'
    """
    nazwa_pliku = os.path.basename(sciezka_pliku)
    # Szukamy tekstu pomiędzy 'results_' a '_final' lub kolejnym podkreśleniem
    match = re.search(r'results_([a-zA-Z0-9]+)', nazwa_pliku)
    if match:
        return match.group(1).upper()
    return "ALPR MODEL"


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

    # --- AUTOMATYCZNE POBRANIE NAZWY MODELU ---
    nazwa_modelu = wyciagnij_nazwe_modelu(sciezka)
    print(f"[+] Wykryty model do wizualizacji: {nazwa_modelu}")

    # Ustawienie globalnego stylu z Seaborn
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 6))  # Lekko poszerzone okno dla lepszej czytelności

    # Dynamiczny tytuł okna systemowego
    fig.canvas.manager.set_window_title(f"Benchmark ALPR - Ewaluacja modelu: {nazwa_modelu}")

    # --- 1. WYKRES: Wydajność Czasowa (Latency) ---
    ax1 = plt.subplot(1, 3, 1)

    yolo_time = dane.get('YOLO_ms', 0.0)
    ocr_time = dane.get('OCR_ms', 0.0)
    e2e_time = dane.get('E2E_ms', 0.0)

    if e2e_time > 0 and yolo_time == 0.0 and ocr_time == 0.0:
        yolo_time = e2e_time * 0.5
        ocr_time = e2e_time * 0.5
        ax1.set_xlabel("*Czasy oszacowane symulacyjnie (50/50)", color="orange")

    czasy = [yolo_time, ocr_time, e2e_time]
    etykiety_czas = [f'Detekcja ({nazwa_modelu})', 'Odczyt OCR', 'Całość E2E']

    bars1 = sns.barplot(x=etykiety_czas, y=czasy, hue=etykiety_czas, palette="rocket", ax=ax1)
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    ax1.set_title(f"Opóźnienie - {nazwa_modelu} [ms]", fontsize=14, pad=15)
    ax1.set_ylabel("Czas [milisekundy]")

    for container in bars1.containers:
        for bar in container:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax1.annotate(f"{h:.1f} ms",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # --- 2. WYKRES: Metryki Jakości + PORÓWNANIE CER ---
    ax2 = plt.subplot(1, 3, 2)
    p_val = dane.get('Precision', 0.0) * 100
    r_val = dane.get('Recall', 0.0) * 100
    f1_val = dane.get('F1', 0.0) * 100
    acc_val = dane.get('Plate_Accuracy', 0.0)

    # Sprawdzamy czy mamy rozbicie na Standardowy i Ważony CER
    has_both_cer = 'Standard_CER' in dane and 'Weighted_CER' in dane

    if has_both_cer:
        std_cer_pct = dane['Standard_CER'] * 100
        wgh_cer_pct = dane['Weighted_CER'] * 100

        metryki_wartosci = [p_val, r_val, f1_val, acc_val, std_cer_pct, wgh_cer_pct]
        etykiety_metryk = ['Precyzja', 'Czułość', 'F1-Score', 'Exact\nMatch', 'Standard\nCER', 'Ważony\nCER']
        paleta = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]
    else:
        legacy_cer = dane.get('CER', 0.0) * 100
        metryki_wartosci = [p_val, r_val, f1_val, acc_val, legacy_cer]
        etykiety_metryk = ['Precyzja', 'Czułość', 'F1-Score', 'Exact Match\n(OCR)', 'Błąd\nCER']
        paleta = "viridis"

    bars2 = sns.barplot(x=etykiety_metryk, y=metryki_wartosci, hue=etykiety_metryk, palette=paleta, ax=ax2)
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    ax2.set_title(f"Metryki Jakości - {nazwa_modelu} (%)", fontsize=14, pad=15)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Skuteczność / Skala Błędu [%]")

    if has_both_cer:
        ax2.get_xticklabels()[-1].set_weight("bold")
        ax2.get_xticklabels()[-1].set_color("#1d71ac")

    for container in bars2.containers:
        for bar in container:
            h = bar.get_height()
            if not np.isnan(h) and h >= 0:
                ax2.annotate(f"{h:.1f}%",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
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

    ax3.set_title(f"Macierz Pomyłek - {nazwa_modelu}", fontsize=14, pad=15)
    ax3.set_xlabel("Przewidywanie modelu", fontsize=12)
    ax3.set_ylabel("Rzeczywistość (Ground Truth)", fontsize=12)
    ax3.set_xticklabels(['Pozytywny', 'Negatywny'])
    ax3.set_yticklabels(['Pozytywny', 'Negatywny'])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rysuj_wykresy()