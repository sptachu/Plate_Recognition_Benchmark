import os
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessor, LogitsProcessorList

# --- USTAWIENIA WIDEO ---
VIDEO_SOURCE = 'testVideo.mp4'
FRAME_SKIP = 5  # Przetwarzaj co 5-tą klatkę

# --- SŁOWNIKI ZNAKÓW CCPD ---
provinces_str = "".join(["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"])
allowed_chars = provinces_str + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- AUTOMATYCZNE WYKRYWANIE SPRZĘTU ---
def detect_hardware():
    if torch.cuda.is_available():
        return 'cuda', True
    return 'cpu', False

# Needed for allowed characters
class TrOCRAllowlistProcessor(LogitsProcessor):
    def __init__(self, allowlist_str, tokenizer):
        self.bad_token_ids = []
        for i in range(tokenizer.vocab_size):
            token_text = tokenizer.decode([i], skip_special_tokens=True).strip()
            if token_text and any(char not in allowlist_str for char in token_text):
                self.bad_token_ids.append(i)
        
        self.mask = torch.zeros(tokenizer.vocab_size)
        self.mask[self.bad_token_ids] = float("-inf")

    def __call__(self, input_ids, scores):
        return scores + self.mask.to(scores.device)


# --- INICJALIZACJA MODELI ---
device_yolo, use_gpu_ocr = detect_hardware()
print(f"[*] Wykryto sprzęt: {device_yolo.upper()}")

print("[*] Ładowanie YOLO11...")
detector = YOLO('yolo11_plate.pt')

print("[*] Ładowanie TrOCR...")
device_ocr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("./Models/trOCR/CCPD")
reader = VisionEncoderDecoderModel.from_pretrained("./Models/trOCR/CCPD")
reader.to(device_ocr)

logits_processor = LogitsProcessorList([
    TrOCRAllowlistProcessor(allowed_chars, processor.tokenizer)
])

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
        
        # Pomijanie klatek w celu optymalizacji
        if frame_count % FRAME_SKIP != 0:
            continue

        h_img, w_img, _ = frame.shape

        # 1. Detekcja YOLO
        results = detector(frame, device=device_yolo, verbose=False)

        for result in results:
            for box in result.boxes:
                # Pobranie koordynatów
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Utrzymana logika odcięcia i paddingu z lewej strony
                oryg_szer = x2 - x1
                odciecie_px = int(oryg_szer * 0.35)
                nx1 = x1 + odciecie_px

                # Wycięcie rejestracji
                plate_crop = frame[max(0, y1):min(h_img, y2), max(0, nx1):min(w_img, x2)]
                
                if plate_crop.size > 0:
                    # Dodanie czarnego paddingu
                    plate_crop = cv2.copyMakeBorder(
                        plate_crop,
                        0, 0, odciecie_px, 0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                    # 2. Rozpoznawanie TrOCR
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(plate_rgb)

                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(device_ocr)
                    generated_ids = reader.generate(pixel_values, max_new_tokens=10)
                    ocr_results = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    read_text = "".join(ocr_results).replace(" ", "").upper()
                    
                    # Wydruk do konsoli
                    print(f"Klatka {frame_count} | Znaleziono blachy na: [{x1}, {y1}, {x2}, {y2}] | OCR: {read_text}")

                    # --- RYSOWANIE NA KLATCE WIDEO ---
                    # Rysuj ramkę wokół wykrytej tablicy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Dodaj tło pod tekst, by był czytelniejszy
                    (text_w, text_h), _ = cv2.getTextSize(read_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                    
                    # Dodaj rozpoznany tekst nad ramką
                    # UWAGA: standardowe fonty cv2 mogą nie renderować chińskich znaków (wyświetlą '?')
                    cv2.putText(frame, read_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Wyświetlanie przetworzonej klatki
        cv2.imshow('Detekcja i OCR Tablic', frame)

        # Wyjście z pętli po wciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Przerwano odtwarzanie przez użytkownika (wciśnięto 'q').")
            break

except KeyboardInterrupt:
    print("\n[!] Przerwano ręcznie przez użytkownika (Ctrl+C).")

finally:
    # Zwalnianie zasobów
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Zakończono.")