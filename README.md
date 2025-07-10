# Soccer Player Re-Identification (Single Feed)

## Overview
This project implements player re-identification in a 15-second football video using a fine-tuned YOLOv11 model for object detection and a simple tracking mechanism.

## Setup
1. Clone the repository: `git clone <repository-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the YOLOv11 model from [Google Drive](https://drive.google.com/file/d/1-5f0SHS0SB9UXyP_en0oZNAMScrEPVcMD/view) and place it in the `models/` folder.
4. Place the input video (`input_720p.mp4`) in the `data/input/` folder.
5. Run the pipeline: `python src/main.py`

## Dependencies
- opencv-python
- torch
- numpy
- json

## Files
- `src/`: Contains all Python scripts.
- `data/`: Stores input video, frames, and labels.
- `models/`: Holds the pre-trained YOLOv11 model.
- `outputs/`: Saves the tracked video and logs.

## Usage
The script processes the video, detects players, assigns IDs in the initial few seconds, and tracks them in real-time.