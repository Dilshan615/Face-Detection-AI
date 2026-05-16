# Face Detection AI (Advanced DNN Version)

A highly accurate Python script that detects human faces in images and draws red rectangles around them. 
This version utilizes OpenCV's Deep Neural Network (DNN) module with a ResNet-10 SSD model, which provides state-of-the-art accuracy, easily detecting side-profiles, small background faces, and avoiding false positives on clothing or hands.

## Features
- **High Accuracy**: Powered by OpenCV's Deep Neural Network (DNN) for robust face detection.
- **GUI Image Selection**: Easily select images from anywhere on your computer using a file dialog.
- **Background Face Detection**: Uses high-resolution scaling to find very small faces in the background.
- **Smart Filtering**: Uses Non-Maximum Suppression (NMS) to ensure each face is only boxed once without duplicates.
- **Auto-Setup**: Automatically downloads the necessary AI model weights on the first run.
- **Image Saving**: Automatically saves the processed image with detected faces highlighted in red.

## Prerequisites

Make sure you have Python installed on your system. You only need the `opencv-python` library.

To install the required library, run the following command in your terminal or command prompt:
```bash
pip install opencv-python
```

*Note: You do NOT need `mediapipe` or any other external libraries. Everything is handled by OpenCV.*

## Project Files
- `face_detector.py`: The main Python script that runs the application.
- `deploy.prototxt`: The AI model configuration file (Required).
- `res10_300x300_ssd_iter_140000.caffemodel`: The pre-trained AI model weights (Required).

## How to Use

1. Open your terminal or command prompt in the project folder.
2. Run the script:
   ```bash
   python face_detector.py
   ```
3. A file selection window will open. Choose any image (jpg, png, etc.) from your computer.
4. The script will process the image, display the result on the screen (resized to fit your monitor), and save a copy of the full-resolution image with the detected faces in the same directory as your original image.

Alternatively, you can provide an image path directly via the terminal:
```bash
python face_detector.py "path/to/your/image.jpg"
```
