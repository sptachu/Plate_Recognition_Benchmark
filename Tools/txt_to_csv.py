import os

# Ścieżki do naszych folderów ze zdjęciami
foldery = ['dataset/UC3M-LP/train_ocr', 'dataset/UC3M-LP/test_ocr']

for folder in foldery:
    txt_path = os.path.join(folder, 'labels.txt')
    csv_path = os.path.join(folder, 'labels.csv')

    if not os.path.exists(txt_path):
        print(f"[!] Brak pliku: {txt_path}")
        continue

    with open(txt_path, 'r', encoding='utf-8') as f:
        linie = f.readlines()

    with open(csv_path, 'w', encoding='utf-8') as f:
        # Ten nagłówek wymusza EasyOCR w nowej wersji
        f.write("filename,words\n")
        for linia in linie:
            if not linia.strip(): continue
            parts = linia.strip().split('\t')
            if len(parts) == 2:
                # Zamiana tabulatora na przecinek
                f.write(f"{parts[0]},{parts[1]}\n")

    print(f"[+] Przekonwertowano {txt_path} -> {csv_path}")