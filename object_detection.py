import os
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

        # --- creative twist: load the sunglasses emoji used to cover phones ---
        # The PNG sits next to this file. cv2 loads it as BGRA; we convert to
        # RGBA because the project pipeline works in RGB. self.emoji stays None
        # if the file is missing, so detection still works without it.
        emoji_path = os.path.join(os.path.dirname(__file__), "sunglasses_emoji.png")
        emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # H x W x 4 (BGRA)
        if emoji is not None and emoji.shape[2] == 4:
            emoji = cv2.cvtColor(emoji, cv2.COLOR_BGRA2RGBA)
        self.emoji = emoji

    def _overlay_emoji(self, frame, x1, y1, x2, y2, scale=1.5):
        '''
        Alpha-blends the sunglasses emoji over the detected box.
        scale > 1 makes it bigger than the box (a "pretty big" emoji).
        Clips to the frame edges so phones near a border don't crash.
        frame is modified in place (RGB uint8).
        '''
        if self.emoji is None:
            return

        h_f, w_f = frame.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1

        # Size the (square) emoji to the larger side of the box, then enlarge.
        size = int(max(box_w, box_h) * scale)
        if size < 1:
            return
        em = cv2.resize(self.emoji, (size, size), interpolation=cv2.INTER_AREA)

        # Center the emoji on the middle of the detected box.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        top, left = cy - size // 2, cx - size // 2

        # Region of the FRAME we will draw onto, clamped to the image bounds.
        y0, y1c = max(0, top), min(h_f, top + size)
        x0, x1c = max(0, left), min(w_f, left + size)
        if y0 >= y1c or x0 >= x1c:
            return  # emoji is entirely off-screen

        # Matching region of the EMOJI (offset by how much we clipped).
        ey0, ex0 = y0 - top, x0 - left
        crop = em[ey0:ey0 + (y1c - y0), ex0:ex0 + (x1c - x0)]

        # Standard alpha compositing: out = a*emoji + (1-a)*background
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        rgb = crop[:, :, :3].astype(np.float32)
        roi = frame[y0:y1c, x0:x1c].astype(np.float32)
        frame[y0:y1c, x0:x1c] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)

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

            if class_name == "cell phone":
                # --- creative twist: cover the phone with a big 'cool' emoji ---
                # instead of a plain rectangle. This is the "image replacement"
                # the task slide asks for, differentiating us from plain-box groups.
                self._overlay_emoji(output_frame, x1, y1, x2, y2, scale=1.5)
            else:
                # Every other class keeps the normal green box + label.
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                text_y = max(y1 - 10, 20)  # keep the label inside the image
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
