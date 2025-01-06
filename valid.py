from ultralytics import YOLO



# Load the YOLO model
model = YOLO("C:/Users/nanda/OneDrive/Desktop/MYproject/runs/detect/train2/weights/best.pt")

# Perform prediction
results = model.predict(source="C:/Users/nanda/OneDrive/Desktop/MYproject/4.jpg", show=True, save=True, conf=0.25, verbose=True)


