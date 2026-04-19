import os
import cv2
import torch
import random
import glob
from ultralytics import YOLO

# --- USTAWIENIA ---
ROOT_DATA_DIR = 'dataset/CCPD2019/'

# ZMIANA: Dokładnie te pliki, które przed chwilą wygenerował Twój pierwszy skrypt!
SPLIT_FILES = {
    'train': 'dataset/CCPD2019/splits/train_nowy.txt',
    'test': 'dataset/CCPD2019/splits/test_nowy.txt'
}

# --- LIMITS ---
# (Zostawiłem na wszelki wypadek, choć listy mają już 9000/1000)
LIMITS = {
    'train': 9000,
    'test': 1000
}

# --- NOWE PARAMETRY CIĘCIA (Pod Fine-Tuning) ---
CROP_LEFT_PERCENT = 0.35  # Odcinamy 37% lewej strony

# --- CCPD DICTIONARIES ---
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def detect_hardware():
    if torch.cuda.is_available(): return 'cuda', True
    return 'cpu', False


def oblicz_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def parse_ccpd_filename(path):
    try:
        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('-')
        if len(parts) < 7: return None, None
        bbox_part = parts[2]
        left_up, right_bottom = bbox_part.split('_')
        x1, y1 = map(int, left_up.split('&'))
        x2, y2 = map(int, right_bottom.split('&'))
        lp_indices = parts[4].split('_')
        prawdziwy_tekst = (
                PROVINCES[int(lp_indices[0])] +
                ALPHABETS[int(lp_indices[1])] +
                "".join([ADS[int(idx)] for idx in lp_indices[2:]])
        ).replace("O", "")
        return [x1, y1, x2, y2], prawdziwy_tekst
    except:
        return None, None


def czysc_folder(dir_path):
    """Usuwa stare zdjęcia przed wygenerowaniem nowych (żeby się nie mieszały)."""
    if os.path.exists(dir_path):
        print(f"[*] Czyszczenie starego folderu: {dir_path}")
        files = glob.glob(os.path.join(dir_path, '*.jpg')) + glob.glob(os.path.join(dir_path, '*.txt'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                pass


def wytnij_i_zapisz():
    device_yolo, _ = detect_hardware()
    detector = YOLO('yolo11_plate.pt')

    for split_name, txt_file in SPLIT_FILES.items():
        if not os.path.exists(txt_file):
            print(f"[!] BŁĄD: Nie znaleziono pliku {txt_file}")
            continue

        output_dir = os.path.join(ROOT_DATA_DIR, f"{split_name}_ocr")
        os.makedirs(output_dir, exist_ok=True)
        czysc_folder(output_dir)  # Odpalenie miotły przed pracą

        labels_file_path = os.path.join(output_dir, 'labels.txt')

        print(f"[*] Wczytywanie listy {split_name}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            target_lines = [line.strip() for line in f if line.strip()]

        saved_crops_count = 0
        with open(labels_file_path, 'w', encoding='utf-8') as labels_file:
            for rel_path in target_lines:
                full_img_path = os.path.join(ROOT_DATA_DIR, rel_path)
                img = cv2.imread(full_img_path)
                if img is None: continue

                h_img, w_img, _ = img.shape
                gt_box, true_txt = parse_ccpd_filename(rel_path)

                if gt_box:
                    results = detector(img, device=device_yolo, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            px1, py1, px2, py2 = map(int, box.xyxy[0])

                            if oblicz_iou([px1, py1, px2, py2], gt_box) >= 0.5:
                                oryg_szer = px2 - px1
                                odciecie_px = int(oryg_szer * CROP_LEFT_PERCENT)
                                nx1 = px1 + odciecie_px

                                # Wycinamy obraz - BEZ PADDINGU (przygotowanie pod FT)
                                plate_crop = img[max(0, py1):min(h_img, py2), max(0, nx1):min(w_img, px2)]

                                if plate_crop.size > 0:
                                    crop_name = f"{split_name}_crop_{saved_crops_count:06d}.jpg"

                                    # Ucinamy prowincję i kropkę z tekstu
                                    czysty_tekst = true_txt[2:]

                                    # Zapisujemy
                                    cv2.imwrite(os.path.join(output_dir, crop_name), plate_crop)
                                    labels_file.write(f"{crop_name}\t{czysty_tekst}\n")
                                    saved_crops_count += 1
                                    break

                if saved_crops_count % 100 == 0:
                    print(f"    Wygenerowano: {saved_crops_count}/{len(target_lines)}", end='\r')

        print(f"\n[+] {split_name} Gotowe! Zapisano {saved_crops_count} sztuk.")


if __name__ == "__main__":
    wytnij_i_zapisz()