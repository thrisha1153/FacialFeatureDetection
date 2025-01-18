import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:\\Users\\thris\\Desktop\\facialfeaturedetector\\best.pt")  # Replace with the actual path

# Open the webcam (usually device ID 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get bounding boxes and labels
    for *xyxy, conf, cls in results[0].boxes.data.tolist():  # Assuming 'results' is a list
        x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
        label = model.names[int(cls)]  # Get the label name

        # Draw bounding boxes and labels on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Green text

    # Display the resulting frame
    cv2.imshow('Webcam Feature Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()