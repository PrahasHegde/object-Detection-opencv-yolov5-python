import cv2
import torch

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = model.autoshape()  # Autoshape input to CUDA FP16

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change the argument if you have multiple cameras (e.g., 1, 2, ...)

# Set the desired window size
window_width = 1280
window_height = 720

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform inference
    results = model(frame)

    # Extract bounding box information
    bboxes = results.xyxy[0].cpu().numpy()

    # Iterate over detected objects
    for bbox in bboxes:
        class_id, confidence, x_min, y_min, x_max, y_max = map(int, bbox[:6])
        label = model.names[0]

        # Draw bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Perform automation based on detected objects
        if label == 'person':
            print("Person detected! Take action here...")

    # Resize the frame to the desired window size
    frame = cv2.resize(frame, (window_width, window_height))

    # Display the frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
