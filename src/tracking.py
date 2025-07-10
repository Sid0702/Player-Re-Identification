import cv2

class Tracker:
    def __init__(self):
        self.trackers = {}

    def initialize_trackers(self, frame, detections, ids):
        for det, id in zip(detections, ids):
            x, y, w, h = det['bbox']
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            self.trackers[id] = tracker

    def update_trackers(self, frame):
        updated_bboxes = {}
        for id, tracker in list(self.trackers.items()):
            success, bbox = tracker.update(frame)
            if success:
                updated_bboxes[id] = bbox
            else:
                del self.trackers[id]  # Remove failed tracker
        return updated_bboxes

if __name__ == "__main__":
    tracker = Tracker()
    frame = cv2.imread("data/frames/frame_001.jpg")
    detections = [{'bbox': (100, 100, 50, 50)}] * 2
    ids = [0, 1]
    tracker.initialize_trackers(frame, detections, ids)