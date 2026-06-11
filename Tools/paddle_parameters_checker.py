import paddle
import numpy as np

sciezka_do_modelu = r'C:\Users\stachu\PycharmProjects\Plate_Recognition_Benchmark\Twoja\Sciezka\Do\inference.pdiparams'

wagi = paddle.load(sciezka_do_modelu)

total_params = 0
for nazwa, macierz in wagi.items():
    if hasattr(macierz, 'size'):
        total_params += macierz.size
    elif isinstance(macierz, np.ndarray):
        total_params += macierz.size

print(f"Liczba parametrów PaddleOCR: {total_params / 1_000_000:.2f} M")