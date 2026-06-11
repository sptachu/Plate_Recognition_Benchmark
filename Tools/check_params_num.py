import torch

# Podaj ścieżkę do swojego wytrenowanego pliku z wagami
sciezka_do_modelu = r'C:\Users\stachu\PycharmProjects\Plate_Recognition_Benchmark\Models\easyOCR\custom_ccpd_easyOCR.pth'

# Ładujemy czysty słownik z wagami (omijamy całą bibliotekę easyocr)
wagi = torch.load(sciezka_do_modelu, map_location='cpu')

# Zliczamy wszystkie parametry z warstw modelu
total_params = sum(p.numel() for p in wagi.values())

print(f"Liczba parametrów w Twoim modelu: {total_params / 1_000_000:.2f} M")


