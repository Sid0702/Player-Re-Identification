import cv2
import os
import torch
from object_detection import load_model, detect_players
from reidentification import ReIdentifier # Ensure this is imported
from my_utils import draw_bboxes # Ensure this is imported

def main():
    # --- Configuration ---
    # Ensure these paths are correct relative to where you run the script
    VIDEO_PATH = "data/input/15sec_input_720p.mp4"
    MODEL_PATH = "models/yolo_v11_model.pt" # Or whatever your YOLO model file is named
    OUTPUT_VIDEO_PATH = "outputs/tracked_video.mp4"
    
    # Use GPU if available, otherwise CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Setup ---
    # Load the detection model
    detection_model = load_model(MODEL_PATH, DEVICE)
    if detection_model is None:
        print("Failed to load detection model. Exiting.")
        return
        
    # Initialize the re-identifier
    re_identifier = ReIdentifier(device=DEVICE, threshold=0.45) # You can adjust the threshold here

    # --- Video I/O ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}. Please check the path and file existence.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create video writer for {OUTPUT_VIDEO_PATH}. Check directory permissions or codec availability.")
        cap.release()
        return

    # --- Main Processing Loop ---
    frame_count = 0
    print("\nStarting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame_count += 1
        # print(f"Processing frame {frame_count}...") # Uncomment for per-frame progress

        # 1. Detect players in the current frame
        detections = detect_players(detection_model, frame, DEVICE)

        # 2. Identify players and assign/re-assign IDs
        # Pass current_frame_idx to ReIdentifier for track management
        # FIX: Added frame_count as the current_frame_idx argument
        bboxes, ids = re_identifier.identify_players(frame, detections, frame_count)

        # 3. Draw bounding boxes and IDs on the frame
        output_frame = draw_bboxes(frame, bboxes, ids)

        # Write frame to output video and display
        out.write(output_frame)
        cv2.imshow("Player Re-Identification", output_frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted processing.")
            break

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    # Ensure the 'outputs' directory exists
    os.makedirs("outputs", exist_ok=True)
    main()
