import os
import cv2
import torch
import numpy as np
import time
import re
import json
import warnings
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Importy z Twojego projektu
from model_lprnet.LPRNet import build_lprnet
from load_data import CHARS

# --- KONFIGURACJA ---
YOLO_PATH = 'yolo11_plate.pt'  # Jeśli nie działa, użyj 'yolov8n.pt' do testu
# W main_lprnet.py zmień:
LPRNET_WEIGHTS = './weights/europe/LPRNet_Euro_Epoch_100.pth'
IMG_SIZE = [94, 24]


class EuropeanALPR:
    def __init__(self, use_cuda=False):
        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

        # 1. Ładowanie YOLO
        print(f"[*] Ładowanie YOLO z {YOLO_PATH}...")
        self.detector = YOLO(YOLO_PATH)

        # 2. Ładowanie LPRNet (Transfer Learning)
        print("[*] Inicjalizacja LPRNet...")
        self.lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(self.device)

        if os.path.exists(LPRNET_WEIGHTS):
            ckpt = torch.load(LPRNET_WEIGHTS, map_location=self.device)
            model_dict = self.lprnet.state_dict()
            # Filtrujemy tylko pasujące warstwy (backbone)
            valid_weights = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(valid_weights)
            self.lprnet.load_state_dict(model_dict)
            self.lprnet.eval()
            print(f"[+] Załadowano {len(valid_weights)} warstw. Gotowy.")
        else:
            print("[!] Brak wag LPRNet!")

    def create_training_data(self, json_dir, images_dir, output_folder):
        """
        Hybrydowe przygotowanie danych:
        - Detekcja: YOLO (widzi tablicę)
        - Etykieta: JSON 'characters' (wie co na niej jest)
        """
        import json
        import re
        import os
        import cv2
        import numpy as np

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print(f"[*] Przetwarzanie {len(json_files)} plików (Detekcja: YOLO, Label: JSON)...")

        count = 0
        for j_file in json_files:
            try:
                # 1. Odczytujemy label (tekst) z JSON
                with open(os.path.join(json_dir, j_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Budujemy etykietę z listy characters
                # Zakładamy pierwszą tablicę z listy [0], bo UC3M ma zazwyczaj jedną główną
                if not data.get('lps'): continue
                chars_list = data['lps'][0].get('characters', [])
                label = "".join([str(c['char_id']) for c in chars_list])
                clean_label = "".join(re.findall(r'[A-Z0-9]', label.upper()))

                if not clean_label: continue

                # 2. Wczytujemy obraz
                img_path = os.path.join(images_dir, data['imagePath'])
                if not os.path.exists(img_path):
                    img_path = os.path.join(images_dir, j_file.replace('.json', '.jpg'))

                img = cv2.imread(img_path)
                if img is None: continue

                # 3. Detekcja YOLO (szukamy tablicy tam, gdzie ona naprawdę jest)
                # conf=0.4, żeby nie łapać śmieci
                results = self.detector.predict(img, conf=0.4, verbose=False)[0]

                if len(results.boxes) == 0:
                    # Jeśli YOLO nic nie widzi, nie mamy co wyciąć
                    continue

                # Bierzemy pierwszą wykrytą ramkę przez YOLO
                box = results.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 4. Wycinanie i skalowanie
                pad = 2
                h_act, w_act = img.shape[:2]
                crop = img[max(0, y1 - pad):min(h_act, y2 + pad),
                       max(0, x1 - pad):min(w_act, x2 + pad)]

                if crop.size == 0: continue

                # Skalujemy do standardu LPRNet
                crop_resized = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_AREA)

                # 5. Zapis: TEKST_ID.jpg
                save_name = f"{clean_label}_{count}.jpg"
                cv2.imwrite(os.path.join(output_folder, save_name), crop_resized)
                count += 1

            except Exception as e:
                print(f"[!] Błąd w {j_file}: {e}")

        print(f"[+] Sukces! YOLO wycięło {count} tablic z etykietami z JSONów.")

    def run_inference(self, img_path):
        """Detekcja i rozpoznawanie"""
        img = cv2.imread(img_path)
        if img is None: return

        # Zwiększamy imgsz jeśli tablice są małe, dodajemy confidence threshold
        results = self.detector.predict(img, conf=0.25, imgsz=640, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # Preprocessing i OCR
            img_res = cv2.resize(crop, (94, 24)).astype('float32')
            img_res = (img_res - 127.5) * 0.0078125
            img_res = np.transpose(img_res, (2, 0, 1))
            tensor = torch.from_numpy(img_res).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.lprnet(tensor)

            # Dekodowanie (uproszczone)
            res_text = self.decode_prediction(logits)
            print(f"--- WYKRYTO: {res_text} ---")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, res_text, (x1, y1 - 10), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Wynik", img)
        cv2.waitKey(0)


import os
import cv2
from ultralytics import YOLO

# Używamy Twojego modelu, który (jak mówisz) jest dobry
model = YOLO('yolo11_plate.pt')


def yolo_only_crop(input_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    print(f"[*] Rozpoczynam wycinanie {len(images)} zdjęć samym YOLO...")
    count = 0

    for img_name in images:
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is None: continue

        # Detekcja (conf=0.4, żeby nie łapać śmieci)
        results = model.predict(img, conf=0.4, verbose=False)[0]

        # Bierzemy nazwę tablicy z nazwy pliku (jeśli tam jest)
        # Np. "KR12345_auto.jpg" -> "KR12345"
        label = img_name.split('_')[0].split('.')[0].upper()

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Ciasne wycięcie
            crop = img[y1:y2, x1:x2]

            if crop.size > 0:
                # Zapisujemy: ETYKIETA_unikalnyID.jpg
                save_path = os.path.join(output_dir, f"{label}_{count}.jpg")
                cv2.imwrite(save_path, crop)
                count += 1

    print(f"[+] Gotowe! Wycięto {count} tablic. Teraz możesz trenować LPRNet.")


# WYWOŁANIE:
# yolo_only_crop("data/UC3M-LP/train/", "data/cropped/")


# --- START ---
if __name__ == "__main__":
    alpr = EuropeanALPR()
    # 1. Wycinanie (użyj tego raz)
    alpr.create_training_data("data/UC3M-LP/train/", "data/UC3M-LP/train/", "data/cropped/")
    # yolo_only_crop("data/UC3M-LP/train/", "data/cropped/")
    # 2. Testowanie (użyj na zdjęciu, którego nie ma w treningu)
    alpr.run_inference("test_auto.jpg")