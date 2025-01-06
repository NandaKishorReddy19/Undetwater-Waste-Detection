from ultralytics import YOLO
model = YOLO("yolo11m.pt")
model.train(data="C:/Users/nanda/OneDrive/Desktop/MYproject/under-water-waste-detection--7/data.yaml", 
            imgsz=640, batch=8, epochs=50, workers=0, device=0)
