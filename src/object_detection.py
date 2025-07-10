import cv2
import torch
from ultralytics import YOLO # Import YOLO from ultralytics

def load_model(model_path, device):
    """Loads the YOLO model using ultralytics.YOLO."""
    try:
        # Load the YOLO model using the ultralytics.YOLO class.
        # This handles model architecture, weights, and device placement automatically.
        model = YOLO(model_path)
        # Ensure the model is on the specified device
        model.to(device)
        print(f"YOLO model loaded successfully from {model_path} and moved to {device}.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and it's a valid Ultralytics YOLO model file.")
        return None

def detect_players(model, frame, device):
    """Detects players in a given frame using the YOLO model."""
    if model is None:
        return []

    detections = []
    
    # Perform inference using the model.predict method.
    # It handles image preprocessing (resizing, normalization) internally.
    # conf: confidence threshold for detections
    # verbose=False: suppresses detailed output for each prediction
    # device: ensures inference runs on the specified device
    results_list = model.predict(source=frame, conf=0.4, verbose=False, device=device)

    # Process results. results_list contains one Results object per image in the batch.
    # Since we pass a single frame, we take the first element.
    if results_list:
        results = results_list[0] # Get the Results object for the current frame

        # Iterate through detected bounding boxes
        if results.boxes is not None:
            for box in results.boxes:
                # Extract bounding box coordinates (x1, y1, x2, y2)
                # .xyxy[0] gives the tensor [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Extract confidence score
                conf = box.conf.item()
                
                # Extract class index
                cls = int(box.cls.item())

                # Check if the detected class is 'player' and confidence is above threshold
                # model.names provides a mapping from class index to class name
                # You might need to adjust 'player' if your model uses a different class name for players
                if 'player' in model.names[cls].lower() and conf > 0.4: # Confidence threshold
                    w, h = x2 - x1, y2 - y1
                    detections.append({'bbox': (x1, y1, w, h), 'confidence': float(conf)})
            
    return detections
