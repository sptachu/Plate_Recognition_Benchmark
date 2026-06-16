import os
import cv2
import torch
from ultralytics import YOLO
import easyocr

# --- USTAWIENIA WIDEO ---
VIDEO_SOURCE = 'testVideo.mp4'
FRAME_SKIP = 5  # Przetwarzaj co 5-tą klatkę

# --- KONFIGURACJA MODELU ---
CUSTOM_MODEL_NAME = 'custom_ccpd_easyOCR'
CUSTOM_MODEL_DIR = './Models/easyOCR'

# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False


# --- INICJALIZACJA MODELI ---
device_yolo, use_gpu_ocr = detect_hardware()
print(f"[*] Wykryto sprzęt: {device_yolo.upper()}")

print("[*] Ładowanie YOLO11...")
detector = YOLO('yolo11_plate.pt')

print("[*] Ładowanie EasyOCR...")
reader = easyocr.Reader(
    lang_list=['en'],  # Alfanumeryczny model dedykowany
    recog_network=CUSTOM_MODEL_NAME,
    user_network_directory=CUSTOM_MODEL_DIR,
    model_storage_directory=CUSTOM_MODEL_DIR,
    gpu=use_gpu_ocr
)

print("\n[*] Rozpoczynam analizę wideo z użyciem EasyOCR...\n")

# --- INICJALIZACJA STRUMIENIA WIDEO ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"[!] Nie można otworzyć wideo: {VIDEO_SOURCE}")
    exit()

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Koniec strumienia wideo.")
            break
            
        frame_count += 1
        
        # Pomijanie klatek w celu optymalizacji wydajności
        if frame_count % FRAME_SKIP != 0:
            continue

        h_img, w_img, _ = frame.shape

        # 1. Detekcja YOLO11
        results = detector(frame, device=device_yolo, verbose=False)

        for result in results:
            for box in result.boxes:
                # Pobranie koordynatów ramki
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                szerokosc = x2 - x1

                # KROK 1: CIĘCIE (Zgodnie z parametrami Twojego treningu: 37% z lewej strony)
                nowe_x1 = int(x1 + (szerokosc * 0.37))
                plate_crop = frame[max(0, y1):min(h_img, y2), max(0, nowe_x1):min(w_img, x2)]

                if plate_crop.size == 0:
                    continue

                # KROK 2: PRE-PROCESSING (Tylko konwersja do odcieni szarości, bez filtrów)
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                # KROK 3: INFERENCJA EASYOCR
                ocr_results = reader.readtext(
                    gray,
                    detail=0,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                )

                # Sklejanie tekstu i czyszczenie spacji
                read_text = "".join(ocr_results).replace(" ", "").upper()
                
                # Logowanie detekcji w terminalu
                print(f"Klatka {frame_count} | Pozycja: [{x1}, {y1}, {x2}, {y2}] | EasyOCR: {read_text}")

                # --- WIZUALIZACJA NA OKIENKU WIDEO ---
                if read_text:
                    # Rysowanie ramki wokół wykrytej tablicy rejestracyjnej
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Tło pod tekst dla poprawy czytelności
                    (text_w, text_h), _ = cv2.getTextSize(read_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (255, 0, 0), -1)
                    
                    # Wypisanie rozpoznanego tekstu (Tylko znaki alfanumeryczne, więc OpenCV wyświetli je czysto)
                    cv2.putText(frame, read_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Wyświetlenie przetworzonej klatki
        cv2.imshow('Detekcja i EasyOCR (Custom Model)', frame)

        # Wyjście z programu po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Przerwano odtwarzanie klatki przez użytkownika ('q').")
            break

except KeyboardInterrupt:
    print("\n[!] Zatrzymano działanie aplikacji poleceniem terminala.")

finally:
    # Czyszczenie zasobów
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Proces zakończony pomyślnie.")