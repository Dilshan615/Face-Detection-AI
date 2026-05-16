import cv2
import sys
import os
import tkinter as tk
from tkinter import filedialog
import urllib.request

def download_model_files():
    # URL paths for the pre-trained OpenCV DNN face detection model
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    # Download the files if they don't exist in the folder yet
    if not os.path.exists(prototxt_path):
        print("Downloading AI model configuration...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
    if not os.path.exists(model_path):
        print("Downloading AI model weights (this may take a few seconds)...")
        urllib.request.urlretrieve(model_url, model_path)
        
    return prototxt_path, model_path

def detect_faces(image_path):
    # Ensure model files are present
    prototxt_path, model_path = download_model_files()
    
    # Load the highly accurate DNN Face Detector from OpenCV
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}. Please check the file path.")
        return

    height, width = img.shape[:2]

    # Prepare the image for the Deep Neural Network (DNN)
    # Using a larger size (800x800) instead of (300x300) helps find smaller faces in the background
    blob = cv2.dnn.blobFromImage(img, 1.0, (800, 800), (104.0, 177.0, 123.0))
    net.setInput(blob)
    
    # Run the detection
    detections = net.forward()
    
    boxes = []
    confidences = []
    
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        
        # Filter out weak detections (0.5 threshold to avoid false positives on background objects)
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure coordinates are within image boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(width - 1, endX)
            endY = min(height - 1, endY)
            
            box_w = endX - startX
            box_h = endY - startY
            
            boxes.append([startX, startY, box_w, box_h])
            confidences.append(confidence)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping duplicate boxes
    # Lowered nms_threshold to 0.2 to merge any boxes that overlap even slightly
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.2)
    
    faces_count = 0
    if len(indices) > 0:
        indices = indices.flatten()
        faces_count = len(indices)
        for i in indices:
            (x, y, w, h) = boxes[i]
            
            # Tighten the bounding box so it focuses strictly on the face (removes neck/hair)
            offset_x = int(w * 0.08)  # 8% from left and right
            offset_y = int(h * 0.12)  # 12% from top and bottom
            
            tight_x = x + offset_x
            tight_y = y + offset_y
            tight_w = w - (2 * offset_x)
            tight_h = h - (2 * offset_y)
            
            # Draw a red rectangle (BGR: 0, 0, 255) around the tightened face area
            cv2.rectangle(img, (tight_x, tight_y), (tight_x+tight_w, tight_y+tight_h), (0, 0, 255), 3)

    # Print the number of detected faces
    print(f"Detected {faces_count} face(s).")
    
    # Create a specific folder to save the detected images
    save_folder = "Detected_Faces"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Save the result image in the new folder
    base_name = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_name)
    output_path = os.path.join(save_folder, f"{filename}_detected{ext}")
    
    cv2.imwrite(output_path, img)
    print(f"Result saved to {os.path.abspath(output_path)}")

    # Display the image (resized to fit the screen)
    display_img = img.copy()
    disp_height, disp_width = display_img.shape[:2]
    max_height = 800
    max_width = 1000
    
    if disp_height > max_height or disp_width > max_width:
        scaling_factor = min(max_width/disp_width, max_height/disp_height)
        new_size = (int(disp_width * scaling_factor), int(disp_height * scaling_factor))
        display_img = cv2.resize(display_img, new_size, interpolation=cv2.INTER_AREA)

    cv2.imshow('Face Detection AI', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = None
    
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        print("Opening file dialog to select an image...")
        root = tk.Tk()
        root.withdraw()
        
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
    
    if image_path:
        detect_faces(image_path)
    else:
        print("No image selected. Please try again.")
