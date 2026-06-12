# Real-Time Computer Vision Virtual Camera

A real-time Computer Vision project that processes webcam frames in Python and streams the processed output through a virtual camera using OBS Studio.

This project was developed as part of a Computer Vision practical course in a group of three students.

## Project Overview

The goal of this project is to build a live video processing pipeline. The system takes frames from a real webcam, applies image processing and object detection methods, and sends the final processed frames to a virtual camera.

The general workflow is:

```text
Webcam Input → Python Processing → Object Detection / Image Processing → Virtual Camera Output → OBS Studio
```

## Features

The project includes several basic image processing operations:

* RGB histogram visualization
* Mean, mode, standard deviation, minimum and maximum values for RGB channels
* Histogram equalization
* Linear brightness/contrast transformation
* Entropy calculation
* Gaussian blur filter
* Sobel edge detection

For the special task, we implemented:

* Real-time object detection using a pre-trained YOLO model
* Bounding boxes around detected objects
* Class labels and confidence scores
* Emoji overlay/image replacement for detected cell phones

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Numba
* pyvirtualcam
* OBS Studio
* Ultralytics YOLO

## Project Structure

```text
VirtualCamera/
│
├── run.py                 # Main file that starts the camera pipeline
├── capturing.py           # Handles webcam input and virtual camera output
├── basics.py              # Basic image processing functions
├── overlays.py            # Draws histogram and text overlays on frames
├── object_detection.py    # YOLO-based object detection and emoji overlay
├── sunglasses_emoji.png   # Emoji image used for phone replacement
├── yolov8n.pt             # Pre-trained YOLO model weights
├── requirements.txt       # Required Python packages
└── README.md
```

## How It Works

The project starts from `run.py`.

First, the webcam is opened through OpenCV. Each video frame is passed into a custom processing function. Inside this function, selected image processing operations are applied, such as equalization, filtering, entropy calculation, and object detection.

The object detection part uses a pre-trained YOLO model. The model analyzes each frame and returns detected objects with their class names, confidence values, and bounding box coordinates. These results are then drawn on the frame using OpenCV.

After processing, the RGB histogram and text information are added as overlays. Finally, the processed frame is sent to a virtual camera using `pyvirtualcam`, where it can be viewed in OBS Studio.

## Installation

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If some packages are not available with the fixed versions, install them manually:

```bash
pip install numpy opencv-python pillow matplotlib numba moviepy keyboard pynput pyvirtualcam ultralytics
```

## Running the Project

Start OBS Studio and make sure the OBS Virtual Camera is enabled.

Then run:

```bash
python run.py
```

Press `q` to stop the camera stream.

## Keyboard Controls

The project supports the following keyboard toggles:

```text
h → Histogram equalization ON/OFF
l → Linear transformation ON/OFF
e → Entropy display ON/OFF
g → Gaussian blur ON/OFF
s → Sobel edge detection ON/OFF
o → Object detection ON/OFF
q → Quit
```

## Example Outputs

Add screenshots here:

```markdown
![Object Detection Example](images/object_detection_example.png)
![Sobel Filter Example](images/sobel_filter_example.png)
```

## Notes

* The YOLO model is used with pre-trained weights because training an object detection model from scratch would require significantly more data and computing power.
* The focus of this project is integrating neural-network-based object detection into a real-time video processing pipeline.
* On macOS, OBS Virtual Camera may need to be enabled under System Settings before running the project.

## Team

Developed by:

* Ali Akbar
* Alnur Nurumov
* Saad Ali Makhdoom
