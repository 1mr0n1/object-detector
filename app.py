from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import io # Needed for reading image from memory

app = Flask(__name__)

# --- Configuration ---
# Using environment variables or defaults for model paths
PROTOTXT_PATH = os.environ.get("PROTOTXT_PATH", "MobileNetSSD_deploy.prototxt.txt")
CAFFEMODEL_PATH = os.environ.get("CAFFEMODEL_PATH", "MobileNetSSD_deploy.caffemodel")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))

# --- Pre-load Model (Load once when the app starts) ---
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "spoon", "chair","apple", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

net = None

def load_model():
    global net
    if not os.path.exists(PROTOTXT_PATH):
        print(f"[ERROR] Prototxt file not found at {PROTOTXT_PATH}")
        return False
    if not os.path.exists(CAFFEMODEL_PATH):
        print(f"[ERROR] Caffemodel file not found at {CAFFEMODEL_PATH}")
        return False
    try:
        print("[INFO] Loading model...")
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("[INFO] Model loaded successfully.")
        return True
    except cv2.error as e:
        print(f"[ERROR] Could not load model. OpenCV error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading the model: {e}")
        return False

# --- Object Detection Function ---
def detect_objects(image_bytes):
    if net is None:
        print("[ERROR] Model not loaded.")
        return None, "Model not loaded"

    try:
        # Read image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Could not decode image"

        (h, w) = image.shape[:2]

        # Preprocess image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # Set input and perform detection
        net.setInput(blob)
        detections = net.forward()

        # CHANGE: Now store details for each detection
        results = []
        # Process detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                idx = int(detections[0, 0, i, 1])
                if idx < len(CLASSES):
                    # Compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Append details to results list
                    label = CLASSES[idx]
                    results.append({
                        "label": label,
                        "confidence": float(confidence), # Ensure JSON serializable
                        "box": [int(startX), int(startY), int(endX), int(endY)] # Ensure JSON serializable
                    })
                else:
                     print(f"[WARNING] Invalid class index {idx} detected, skipping.")

        # return detected_objects, None # Old way
        return results, None # Return the detailed list

    except Exception as e:
        print(f"[ERROR] Error during object detection: {e}")
        return None, f"Error during processing: {e}"

# --- Routes ---
@app.route("/")
def serve_index():
    """Serves the index.html file."""
    # Serve index.html from the current directory ('.')
    return send_from_directory('.', 'index.html')

@app.route("/image-upload", methods=["POST"])
def handle_image_upload():
    """Accepts an image file upload and returns detected objects."""
    if net is None:
        return jsonify({"error": "Model is not loaded or failed to load."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read image file bytes
        img_bytes = file.read()

        # Perform detection
        print("[DEBUG] Calling detect_objects...")
        detection_results, error_msg = detect_objects(img_bytes)
        print(f"[DEBUG] detect_objects returned: results={detection_results}, error='{error_msg}'")

        if error_msg:
            response_data = {"error": error_msg}
            print(f"[DEBUG] Returning error JSON: {response_data}")
            return jsonify(response_data), 500

        # CHANGE: Return the detailed detection results
        response_data = {"detections": detection_results}
        print(f"[DEBUG] Returning success JSON: {response_data}")
        return jsonify(response_data)

    # This part should ideally not be reached if file checks are robust
    response_data = {"error": "An unknown error occurred after file handling."}
    print(f"[DEBUG] Returning fallback error JSON: {response_data}")
    return jsonify(response_data), 500

# --- Main Execution ---
if __name__ == "__main__":
    if load_model(): # Try to load the model on startup
        # Run the Flask app
        # Use host='0.0.0.0' to make it accessible on your network
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("[FATAL] Could not load the model. Exiting.") 