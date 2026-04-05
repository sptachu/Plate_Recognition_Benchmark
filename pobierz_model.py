import urllib.request

# Link do udostępnionego, wytrenowanego modelu YOLO11n (wersja Nano - ultraszybka) na tablice
url = "https://huggingface.co/DFK1991/yolov11n_car_plates_detector/resolve/main/best_n.pt"
nazwa_pliku = "yolo11_plate.pt"

print(f"Rozpoczynam pobieranie modelu YOLO11... Proszę czekać.")

try:
    # Przedstawiamy się jako przeglądarka, by uniknąć blokady 401
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

    with urllib.request.urlopen(req) as response, open(nazwa_pliku, 'wb') as out_file:
        out_file.write(response.read())

    print(f"Sukces! Plik '{nazwa_pliku}' (YOLO11) został pomyślnie pobrany do folderu projektu.")
except Exception as e:
    print(f"Wystąpił błąd podczas pobierania: {e}")