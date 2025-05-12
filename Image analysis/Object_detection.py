from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv5 model
model = YOLO("yolov8m.pt")



# Run inference
results = model('demo_img.jpeg', conf=0.25)

texts = []
for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())  # Get class ID
        class_name = model.names[class_id]  # Convert class ID to label
        texts.append(class_name)  # Format as a descriptive phrase

# Print the formatted text array
print(list(set(texts)))
