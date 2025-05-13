from ultralytics import YOLO
import cv2

# Load YOLOv8 model once
model = YOLO("yolov8l.pt")

def detect_objects(image_path, confidence=0.25, show=False):
    """
    Runs YOLO object detection on the given image.

    Args:
        image_path (str): Path to the image file.
        confidence (float): Confidence threshold for detection.
        show (bool): Whether to display the image with bounding boxes.

    Returns:
        list: A list of detected object names.
    """
    results = model(image_path, conf=confidence)

    detected_objects = set()  # Use a set to store unique objects
    img = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            detected_objects.add(class_name)

            # Draw bounding box if show=True
            if show:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with detections (for debugging)
    if show:
        cv2.imshow("Detected Objects", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return list(detected_objects)  # Return unique detected objects
