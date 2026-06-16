import cv2
import hyperlpr3 as lpr3

# --- USTAWIENIA WIDEO ---
VIDEO_SOURCE = 'testVideo.mp4'
FRAME_SKIP = 5  # Przetwarzaj co 5-tą klatkę

# --- INICJALIZACJA HYPERLPR3 ---
print("[*] Inicjalizacja HyperLPR3 (Wysoki poziom detekcji)...")
# DETECT_LEVEL_HIGH zapewnia dokładniejsze dopasowanie na poruszającym się wideo
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

print("\n[*] Rozpoczynam analizę strumienia wideo...\n")

# --- INICJALIZACJA STRUMIENIA WIDEO ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"[!] Nie można otworzyć źródła wideo: {VIDEO_SOURCE}")
    exit()

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Strumień wideo zakończył się lub wystąpił błąd odczytu.")
            break
            
        frame_count += 1
        
        # Pomijanie klatek w celu zachowania płynności (Real-Time Performance)
        if frame_count % FRAME_SKIP != 0:
            continue

        # 1. Wywołanie end-to-end silnika HyperLPR3 na całej klatce obrazu
        results = catcher(frame)

        # 2. Przetwarzanie spakowanych wyników wejściowych
        for res_text, conf, p_type, p_box in results:
            # Rozpakowanie współrzędnych [x1, y1, x2, y2] przekazanych przez model
            x1, y1, x2, y2 = map(int, p_box)
            pred_text = res_text.replace(" ", "")

            # Logowanie odczytu bezpośrednio w konsoli systemowej
            print(f"Klatka {frame_count} | Tablica na: [{x1}, {y1}, {x2}, {y2}] | HyperLPR3: {pred_text} (Pewność: {conf:.2f})")

            # --- RYSOWANIE WYNIKÓW NA EKRANIE ---
            # Rysowanie ramki ograniczającej (Bounding Box) wokół tablicy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tworzenie wypełnionego paska pod napisy dla lepszego kontrastu
            (text_w, text_h), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Wyświetlenie odczytanego ciągu znaków nad ramką rejestracji
            # (Uwaga: Czcionka OpenCV wyświetli prowincje CCPD jako znaki zapytania '?')
            cv2.putText(frame, pred_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Wyświetlanie przetworzonego obrazu w dedykowanym GUI
        cv2.imshow('Analiza Wideo - HyperLPR3 End-to-End', frame)

        # Bezpieczne zamknięcie okna za pomocą klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Wyświetlanie przerwane przez użytkownika.")
            break

except KeyboardInterrupt:
    print("\n[!] Działanie aplikacji przerwane z poziomu terminala (Ctrl+C).")

finally:
    # Zwalnianie wątków systemowych i zamykanie instancji okien
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Zamknięto strumienie, program zakończony.")