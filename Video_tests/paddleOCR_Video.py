import os
import cv2
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- USTAWIENIA WIDEO ---
VIDEO_SOURCE = 'testVideo.mp4'
FRAME_SKIP = 5  # Przetwarzaj co 5-tą klatkę

# --- SŁOWNIKI ZNAKÓW CCPD ---
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False

def postprocess_plate(raw_text, target_len=5):
    """ Cleans and forces the OCR output to adhere to the masked CCPD plate format. """
    text = ''.join([c for c in raw_text if c.isalnum()]).upper()

    if len(text) > target_len:
        text = text[:target_len] 

    char_list = list(text)
    forbidden_to_num = {'I': '1', 'O': '0'} 

    for i in range(0, len(char_list)):
        if char_list[i] in forbidden_to_num:
            char_list[i] = forbidden_to_num[char_list[i]]
            
    return "".join(char_list)


# --- INICJALIZACJA MODELI ---
device_yolo, use_gpu_ocr = detect_hardware()
print(f"[*] Wykryto sprzęt: {device_yolo.upper()}")

print("[*] Ładowanie YOLO11...")
detector = YOLO('yolo11_plate.pt')

print("[*] Ładowanie PaddleOCR...")
# Wskazówka: Możesz zmienić use_gpu=use_gpu_ocr, aby automatycznie przyspieszyć działanie na karcie CUDA!
reader = PaddleOCR(lang="ch", use_angle_cls=False, use_gpu=use_gpu_ocr, show_log=False)

print("\n[*] Rozpoczynam analizę wideo...\n")

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
            print("[*] Koniec wideo (lub błąd odczytu).")
            break
            
        frame_count += 1
        
        # Pomijanie klatek w celu optymalizacji wydajności
        if frame_count % FRAME_SKIP != 0:
            continue

        h_img, w_img, _ = frame.shape

        # 1. Detekcja YOLO
        results = detector(frame, device=device_yolo, verbose=False)

        for result in results:
            for box in result.boxes:
                # Pobranie koordynatów ramki tablicy
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                oryg_szer = x2 - x1
                odciecie_px = int(oryg_szer * 0.35)
                nx1 = x1 + odciecie_px

                # Wycięcie interesującego nas fragmentu
                plate_crop = frame[max(0, y1):min(h_img, y2), max(0, nx1):min(w_img, x2)]
                
                if plate_crop.size > 0:
                    # Zastosowanie ulepszenia CLAHE (zgodnie z oryginalnym kodem)
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced_gray = clahe.apply(gray)
                    plate_crop = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

                    # Dodanie czarnego paddingu z lewej strony
                    plate_crop = cv2.copyMakeBorder(
                        plate_crop,
                        0, 0, odciecie_px, 0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                    if plate_crop.size == 0:
                        continue
                    
                    # Ręczne wymuszenie proporcji dla silnika PaddleOCR
                    plate_resized = cv2.resize(plate_crop, (320, 48))

                    # 2. Rozpoznawanie tekstów przez PaddleOCR
                    ocr_results = reader.ocr(plate_resized, det=False, cls=False)

                    ocr_results_str = ""
                    if ocr_results and isinstance(ocr_results[0], list):
                        ocr_results_str = ocr_results[0][0][0]

                    # Heurystyka i czyszczenie tekstu
                    read_text = postprocess_plate(ocr_results_str, target_len=5)
                    
                    # Logowanie wyników w konsoli
                    print(f"Klatka {frame_count} | Tablica na: [{x1}, {y1}, {x2}, {y2}] | PaddleOCR: {read_text}")

                    # --- WIZUALIZACJA NA OKIENKU WIDEO ---
                    # Rysowanie zielonej ramki detektora
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Tło pod napisy (zwiększa czytelność tekstu)
                    (text_w, text_h), _ = cv2.getTextSize(read_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                    
                    # Nałożenie tekstu
                    cv2.putText(frame, read_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Wyświetlanie aktualnej klatki wideo
        cv2.imshow('Detekcja i PaddleOCR Tablic', frame)

        # Wyjście z programu za pomocą klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Przerwano odtwarzanie przez użytkownika (wciśnięto 'q').")
            break

except KeyboardInterrupt:
    print("\n[!] Przerwano ręcznie przez użytkownika (Ctrl+C).")

finally:
    # Sprzątanie zasobów systemowych
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Zakończono pomyślnie.")