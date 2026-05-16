# Face Detection AI

A highly accurate Python script that detects human faces in images and draws red rectangles around them. 
This version utilizes OpenCV's Deep Neural Network (DNN) module with a ResNet-10 SSD model, which provides state-of-the-art accuracy, easily detecting side-profiles, small background faces, and avoiding false positives on clothing or hands.

## ✨ Features
- **High Accuracy**: Powered by OpenCV's Deep Neural Network (DNN) for robust face detection.
- **GUI Image Selection**: Easily select images from anywhere on your computer using a built-in file dialog.
- **Background Face Detection**: Uses high-resolution scaling to find very small faces in the background.
- **Smart Filtering**: Uses Non-Maximum Suppression (NMS) to ensure each face is only boxed once without duplicates.
- **Precision Bounding Boxes**: Automatically tightens bounding boxes to strictly frame facial features (excluding neck and hair).
- **Auto-Setup**: Automatically downloads the necessary AI model weights on the first run.
- **Organized Outputs**: Automatically creates a `Detected_Faces` directory and saves all processed images there, keeping your workspace clean.

## 🛠️ Prerequisites

Make sure you have Python installed on your system. You only need the `opencv-python` library to run this script.

To install the required library, run the following command in your terminal or command prompt:
```bash
pip install opencv-python
```

*Note: You do NOT need `mediapipe` or any other external libraries. Everything is handled directly by OpenCV.*

## 📂 Required AI Model Files

For the AI to work, it requires two specific model files. The Python script is programmed to **automatically download these files** on its first run. However, if you want to download them manually, you can get them from the official OpenCV repository:

1. **Model Configuration File** (`deploy.prototxt`)
   - [Download deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
2. **Model Weights** (`res10_300x300_ssd_iter_140000.caffemodel`)
   - [Download res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

*If downloading manually, place both files in the same directory as `face_detector.py`.*

## 🚀 How to Use

1. Open your terminal or command prompt in the project folder.
2. Run the script without any arguments to open the file selection window:
   ```bash
   python face_detector.py
   ```
3. Choose any image (`.jpg`, `.png`, etc.) from your computer.
4. The script will process the image and display the result on your screen (resized to fit your monitor).
5. A copy of the full-resolution image with the detected faces will be saved securely inside the `Detected_Faces` folder within your project directory.

Alternatively, you can provide an image path directly via the terminal:
```bash
python face_detector.py "path/to/your/image.jpg"
```
