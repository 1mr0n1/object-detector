import cv2
import numpy as np
import argparse
import os

# --- Argument Parsing ---
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    return args

args = parse_arguments()

# --- Configuration ---
# Get values from command line arguments
prototxt_path = args["prototxt"]
caffemodel_path = args["model"]
confidence_threshold = args["confidence"]
image_path = args["image"] # Get image path from args

# Check if model files exist
if not os.path.exists(prototxt_path):
    print(f"[ERROR] Prototxt file not found at {prototxt_path}")
    exit()
if not os.path.exists(caffemodel_path):
    print(f"[ERROR] Caffemodel file not found at {caffemodel_path}")
    exit()

# Class labels the model was trained on (MobileNet SSD with COCO dataset)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign random colors to each class for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# --- Load Model ---
print("[INFO] Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
except cv2.error as e:
    print(f"[ERROR] Could not load model from {prototxt_path} and {caffemodel_path}. OpenCV error: {e}")
    exit()
print("[INFO] Model loaded successfully.")

# --- Load Image ---
# Check if image file exists before trying to read
if not os.path.exists(image_path):
    print(f"[ERROR] Image file not found at {image_path}")
    exit()

try:
    image = cv2.imread(image_path)
    if image is None:
        # This specific check might be redundant now with os.path.exists,
        # but imread can return None for other reasons (e.g., corrupted file)
        raise IOError(f"Could not read image file: {image_path} (possibly corrupted or unsupported format)")
    (h, w) = image.shape[:2]
except IOError as e:
    print(f"[ERROR] {e}")
    exit()
except Exception as e:
    print(f"[ERROR] An unexpected error occurred loading the image: {e}")
    exit()

print(f"[INFO] Loaded image {image_path} with shape: ({h}, {w})")

# --- Preprocess Image and Detect Objects ---
print("[INFO] Preprocessing image and running detection...")
# Create a blob from the image (resize to 300x300, mean subtraction)
# MobileNet SSD expects 300x300 inputs
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Set the blob as input to the network
net.setInput(blob)

# Perform forward pass and get detections
detections = net.forward()

print(f"[INFO] Found {detections.shape[2]} potential detections.")

# --- Process Detections and Draw Bounding Boxes ---
print("[INFO] Processing detections...")
count = 0
# Loop over the detections
# detections shape: (1, 1, N, 7), where N is the number of detections
# Each detection: [batchId, classId, confidence, left, top, right, bottom]
for i in np.arange(0, detections.shape[2]):
    # Extract the confidence (probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence threshold
    if confidence > confidence_threshold:
        count += 1
        # Extract the index of the class label from the `detections`
        idx = int(detections[0, 0, i, 1])

        # Check if the detected class index is valid
        if idx >= len(CLASSES):
            print(f"[WARNING] Invalid class index {idx} detected, skipping.")
            continue

        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Display the prediction
        label = f"{CLASSES[idx]}: {confidence:.2f}"
        print(f"[INFO] Detected: {label} at Box: ({startX},{startY})-({endX},{endY})")
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

print(f"[INFO] Drew {count} boxes with confidence > {confidence_threshold}")

# --- Display Output ---
print("[INFO] Displaying output image. Press any key to exit.")
cv2.imshow("Object Detection Output", image)
cv2.waitKey(0) # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
print("[INFO] Finished.") 