from ultralytics import YOLO

# Ładujemy model przez natywną bibliotekę YOLO
model = YOLO(r'C:\Users\stachu\PycharmProjects\Plate_Recognition_Benchmark\yolo11_plate.pt')

# Ta jedna funkcja wypluje Ci gotowe inżynierskie podsumowanie
model.info()