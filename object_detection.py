import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        # Load the pre-trained YOLO model once
        self.model = YOLO(model_path)

        # Minimum confidence needed to show a detection
        self.confidence = confidence

        # Class names, for example: person, chair, bottle, laptop, etc.
        self.class_names = self.model.names


    def detect_objects(self, frame_rgb):
        # Make sure the image is stored properly in memory for OpenCV
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # YOLO/OpenCV usually work with BGR images, but our project uses RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Run object detection on the current frame
        results = self.model(frame_bgr, conf=self.confidence, imgsz=320, verbose=False)

        # Take the first result because we only passed one image/frame
        result = results[0]

        # Copy the frame so we can draw boxes on it
        output_frame = frame_rgb.copy()

        # Counter for how many objects were detected
        object_count = 0

        # If no boxes are found, return the original frame
        if result.boxes is None:
            return output_frame, object_count

        # Go through every detected object
        for box in result.boxes:
            # Confidence score of this detection
            conf = float(box.conf[0])

            # Class ID, for example 0 might be "person"
            class_id = int(box.cls[0])

            # Get the class name from the class ID
            class_name = self.class_names[class_id]

            # Get bounding box coordinates: top-left and bottom-right
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Draw a rectangle around the detected object
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Text that will be shown above the box
            label = f"{class_name} {conf:.2f}"

            # Make sure text does not go outside the image
            text_y = max(y1 - 10, 20)

            # Draw the label text
            cv2.putText(
                output_frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            object_count += 1

        return output_frame, object_count