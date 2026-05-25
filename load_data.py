from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os


# CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',

#          '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',

#          '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',

#          '新',

#          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

#          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',

#          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',

#          'W', 'X', 'Y', 'Z', 'I', 'O', '-'

#          ]

# Słownik europejski
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z',
    '-'  # Znak 'blank' dla CTC Loss
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            # paths.list_images ładuje wszystkie zdjęcia z podanego folderu
            self.img_paths += [el for el in paths.list_images(img_dir[i])]

        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]

        # Bezpieczne wczytywanie obrazu
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            # Jeśli plik jest uszkodzony, spróbuj wczytać następny
            return self.__getitem__((index + 1) % len(self.img_paths))

        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)

        image = self.PreprocFun(image)

        # --- LOGIKA WYCIĄGANIA ETYKIETY ---
        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)

        # Wyciągamy część przed pierwszym "_" (zgodnie z naszym skryptem wycinającym)
        imgname = imgname.split("_")[0].upper()

        label = list()
        for c in imgname:
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            # Ignorujemy znaki, których nie ma w słowniku, aby uniknąć KeyError

        # UWAGA: Usunięto funkcję self.check(), bo blokowała europejskie tablice!

        return image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img