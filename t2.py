from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import easyocr
from collections import Counter
import re

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("/Users/thehetshah/Desktop/basketball/video/basket.mp4")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
ocr_reader = easyocr.Reader(['en'], gpu=False)
TARGET_CLASSES = [0, 32]

jersey_map = {}       # track_id -> stable jersey number string
jersey_history = {}   # track_id -> list of last 3 jersey reads

# Adjust this ROI to your basket location
basket_roi = (550, 50, 630, 130)  # (x1, y1, x2, y2)

def point_in_roi(x, y, roi):
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2

pass_count = 0
basket_count = 0
prev_ball_possessor = None
ball_in_basket = False

# Print frame shape to help adjust ROI
ret, frame = cap.read()
if ret:
    print("Frame size (h, w, c):", frame.shape)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

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

    ball_detections = [(xyxy, tid) for xyxy, cls_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id) if cls_id == 32]
    player_detections = [(xyxy, tid) for xyxy, cls_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id) if cls_id == 0]

    ball_possessor = None

    for xyxy, class_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)

        if class_id == 0:  # person
            jersey_number = jersey_map.get(track_id)
            if jersey_number is None:
                # Cropped ROI: upper-middle torso region
                y_start = y1 + int((y2 - y1) * 0.35)
                y_end = y1 + int((y2 - y1) * 0.65)
                jersey_crop = frame[y_start:y_end, x1:x2]

                if jersey_crop.size != 0:
                    # Convert to grayscale
                    gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)

                    # Light Gaussian blur to reduce noise
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

                    # Threshold to isolate numbers (light invert)
                    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)

                    # OPTIONAL: Save intermediate crops for debugging
                    # cv2.imwrite(f"debug_crop_{track_id}.png", thresh)

                    ocr_results = ocr_reader.readtext(thresh)

                    candidate_numbers = []

                    for _, text, conf in ocr_results:
                        if conf > 0.5:
                            digits = re.findall(r'\d+', text)
                            for d in digits:
                                if 0 < int(d) < 100:
                                    candidate_numbers.append(d)

                    if candidate_numbers:
                        jersey_history.setdefault(track_id, []).extend(candidate_numbers)
                        jersey_history[track_id] = jersey_history[track_id][-3:]
                        most_common = Counter(jersey_history[track_id]).most_common(1)
                        jersey_number = most_common[0][0]
                        jersey_map[track_id] = jersey_number


            # if jersey_number is None:
            #     y_start = y1 + int((y2 - y1) * 0.3)
            #     y_end = y1 + int((y2 - y1) * 0.7)
            #     jersey_crop = frame[y_start:y_end, x1:x2]

            #     if jersey_crop.size != 0:
            #         jersey_crop_gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
            #         _, jersey_crop_thresh = cv2.threshold(jersey_crop_gray, 150, 255, cv2.THRESH_BINARY_INV)

            #         ocr_results = ocr_reader.readtext(jersey_crop_thresh)

            #         candidate_numbers = []

            #         for _, text, conf in ocr_results:
            #             cleaned = ''.join(filter(str.isdigit, text))
            #             if cleaned and len(cleaned) <= 2 and conf > 0.5:
            #                 number = int(cleaned)
            #                 if 0 < number < 100:
            #                     candidate_numbers.append(str(number))

            #         if candidate_numbers:
            #             jersey_history.setdefault(track_id, []).extend(candidate_numbers)
            #             jersey_history[track_id] = jersey_history[track_id][-3:]
            #             most_common = Counter(jersey_history[track_id]).most_common(1)
            #             jersey_number = most_common[0][0]
            #             jersey_map[track_id] = jersey_number

            label = f"#{track_id} Jersey {jersey_number}" if jersey_number else f"#{track_id} Player"

        elif class_id == 32:  # ball
            label = f"#{track_id} Ball"
        else:
            label = f"#{track_id}"

        labels.append(label)

    # Pass and basket counting logic
    if ball_detections:
        ball_xyxy, ball_tid = ball_detections[0]
        bx1, by1, bx2, by2 = map(int, ball_xyxy)
        ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

        min_dist = float('inf')
        for (px1, py1, px2, py2), player_tid in player_detections:
            px1, py1, px2, py2 = map(int, (px1, py1, px2, py2))
            player_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            dist = (player_center[0] - ball_center[0]) ** 2 + (player_center[1] - ball_center[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                ball_possessor = player_tid

        if point_in_roi(*ball_center, basket_roi):
            if not ball_in_basket:
                basket_count += 1
                ball_in_basket = True
                print(f"Basket made! Count: {basket_count}")  # <-- Debug print
        else:
            ball_in_basket = False
    else:
        ball_possessor = None
        ball_in_basket = False

    if prev_ball_possessor is not None and ball_possessor is not None:
        if ball_possessor != prev_ball_possessor:
            pass_count += 1
            print(f"Pass detected! Count: {pass_count}")  # Optional debug print

    prev_ball_possessor = ball_possessor

    if len(labels) == len(tracked):
        tracked.labels = labels
    else:
        tracked.labels = [f"#{track_id}" for track_id in tracked.tracker_id]

    annotated_frame = box_annotator.annotate(scene=frame, detections=tracked)

    for (xyxy, label) in zip(tracked.xyxy, labels):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(annotated_frame, (basket_roi[0], basket_roi[1]), (basket_roi[2], basket_roi[3]), (0, 255, 255), 2)

    cv2.putText(annotated_frame, f"Passes: {pass_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Baskets Made: {basket_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Basketball Tracking + Jersey OCR + Stats", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
