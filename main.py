from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import easyocr
from collections import Counter

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("/Users/thehetshah/Desktop/basketball/video/basket.mp4")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
ocr_reader = easyocr.Reader(['en'], gpu=False)
TARGET_CLASSES = [0, 32]

jersey_map = {}       # track_id -> stable jersey number string
jersey_history = {}   # track_id -> list of last 3 jersey reads

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    mask = np.isin(detections.class_id, TARGET_CLASSES)
    detections = detections[mask]
    tracked = tracker.update_with_detections(detections)

    labels = []

    for xyxy, class_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)

        if class_id == 0:  # person
            jersey_number = jersey_map.get(track_id)

            if jersey_number is None:
                y_start = y1 + int((y2 - y1) * 0.3)
                y_end = y1 + int((y2 - y1) * 0.7)
                jersey_crop = frame[y_start:y_end, x1:x2]

                if jersey_crop.size != 0:
                    # Preprocess for OCR
                    jersey_crop_gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
                    _, jersey_crop_thresh = cv2.threshold(jersey_crop_gray, 150, 255, cv2.THRESH_BINARY_INV)

                    ocr_results = ocr_reader.readtext(jersey_crop_thresh)

                    candidate_numbers = []

                    for _, text, conf in ocr_results:
                        cleaned = ''.join(filter(str.isdigit, text))
                        if cleaned and len(cleaned) <= 2 and conf > 0.5:
                            number = int(cleaned)
                            if 0 < number < 100:
                                candidate_numbers.append(str(number))

                    if candidate_numbers:
                        # Append to history
                        jersey_history.setdefault(track_id, []).extend(candidate_numbers)
                        # Keep last 3
                        jersey_history[track_id] = jersey_history[track_id][-3:]
                        # Stable number: most common in last 3
                        most_common = Counter(jersey_history[track_id]).most_common(1)
                        jersey_number = most_common[0][0]
                        jersey_map[track_id] = jersey_number

            label = f"#{track_id} Jersey {jersey_number}" if jersey_number else f"#{track_id} Player"

        elif class_id == 32:  # ball
            label = f"#{track_id} Ball"
        else:
            label = f"#{track_id}"

        labels.append(label)

    if len(labels) == len(tracked):
        tracked.labels = labels
    else:
        tracked.labels = [f"#{track_id}" for track_id in tracked.tracker_id]

    annotated_frame = box_annotator.annotate(scene=frame, detections=tracked)

    # Manual label drawing for visibility
    for (xyxy, label) in zip(tracked.xyxy, labels):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Basketball Tracking + Jersey OCR", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
