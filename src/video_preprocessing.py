import cv2
import os

def extract_frames(video_path, output_dir, interval=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "data/input/15sec_input_720p.mp4"
    output_dir = "data/frames/"
    extract_frames(video_path, output_dir)