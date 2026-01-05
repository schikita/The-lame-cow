import cv2 
from ultralytics import YOLO 
import supervision as sv 
import time

model = YOLO("app.pt")

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fps_counter = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id} {model.model.names[class_id]}"
        for class_id, tracker_id
        in zip(tracked_detections.class_id, tracked_detections.tracker_id)
    ]
    
    annotated_frame = frame.copy()
    
    if len(tracked_detections) > 0:
        if tracked_detections.mask is not None:
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections
            )
        
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections
        )
        
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections,
            labels=labels
        )
    
    fps_counter += 1
    if fps_counter % 30 == 0:
        elapsed = time.time() - start_time
        fps = 30 / elapsed
        print(f"FPS: {fps:.1f}, Objects: {len(tracked_detections)}")
        start_time = time.time()
    
    cv2.imshow("YOLO Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()