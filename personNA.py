import cv2
import torch
from age_gender_estimation.age_gender_estimator import AgeGenderEstimator

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov8s')

# Load Age-Gender Estimation model
age_gender_estimator = AgeGenderEstimator()

# Define webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make predictions
    results = model(frame)

    # Extract bounding boxes and labels
    boxes = results.xyxy[0].cpu().numpy()
    labels = results.names[0].cpu().numpy()

    # Draw bounding boxes and labels on the frame
    for box, label in zip(boxes, labels):
        if label == 'person':
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract face from bounding box
            face = frame[y1:y2, x1:x2]

            # Estimate age and gender
            age, gender = age_gender_estimator.predict_age_and_gender(face)

            # Draw age and gender on the frame
            cv2.putText(frame, f"{label} - {age} - {gender}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
