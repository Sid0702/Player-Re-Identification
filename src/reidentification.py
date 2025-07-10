import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment # For optimal matching

class ReIdentifier:
    def __init__(self, device, threshold=0.45, max_frames_unseen=5):
        """
        Initializes the ReIdentifier.

        Args:
            device (torch.device): The device (CPU or CUDA) to run the model on.
            threshold (float): Cosine similarity threshold for matching existing tracks.
            max_frames_unseen (int): Maximum number of frames a track can be unseen before termination.
        """
        self.device = device
        self.threshold = threshold
        self.max_frames_unseen = max_frames_unseen
        self.tracks = {}  # Stores active tracks: {id: {'bbox': (x,y,w,h), 'feature': tensor, 'last_seen_frame': int}}
        self.next_id = 0  # Counter for assigning new unique IDs

        # Load a pre-trained ResNet50 model for feature extraction
        # We use ResNet50_Weights.IMAGENET1K_V1 for consistency with common practices
        # and to avoid the deprecation warning for 'pretrained=True'.
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get feature vectors
        self.feature_extractor = nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval() # Set to evaluation mode

        # Define image transformations for feature extraction
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # ResNet expects 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"Re-identification model (ResNet50) loaded successfully on {self.device}.")

    def _extract_features(self, frame, bbox):
        """
        Extracts features from a cropped bounding box in the frame.

        Args:
            frame (np.array): The full video frame (BGR).
            bbox (tuple): Bounding box in (x, y, w, h) format.

        Returns:
            torch.Tensor: Feature vector for the cropped region.
        """
        x, y, w, h = bbox
        # Ensure bounding box coordinates are within frame boundaries
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        if x2 <= x1 or y2 <= y1: # Handle invalid or empty bounding boxes
            return None

        # Crop the player region from the frame
        cropped_img = frame[y1:y2, x1:x2]
        
        # Ensure the cropped image is not empty
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            return None

        # Convert BGR (OpenCV default) to RGB
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for torchvision transforms
        pil_img = Image.fromarray(cropped_img_rgb)

        # Apply transformations and extract features
        with torch.no_grad():
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            features = self.feature_extractor(img_tensor).squeeze()
        return features / features.norm() # Normalize features

    def _calculate_cosine_similarity(self, feature1, feature2):
        """
        Calculates cosine similarity between two feature vectors.

        Args:
            feature1 (torch.Tensor): First feature vector.
            feature2 (torch.Tensor): Second feature vector.

        Returns:
            float: Cosine similarity.
        """
        if feature1 is None or feature2 is None:
            return 0.0
        return torch.nn.functional.cosine_similarity(feature1, feature2, dim=0).item()

    def identify_players(self, frame, detections, current_frame_idx):
        """
        Identifies players in the current frame and assigns/re-assigns IDs.

        Args:
            frame (np.array): The current video frame.
            detections (list): List of dictionaries, each with 'bbox' (x,y,w,h) and 'confidence'.
            current_frame_idx (int): The index of the current frame being processed.

        Returns:
            tuple: (list of bboxes, list of IDs) for the identified players in the current frame.
        """
        current_bboxes = []
        current_ids = []
        
        # Extract features for all current detections
        detection_features = []
        valid_detections = []
        for det in detections:
            bbox = det['bbox']
            features = self._extract_features(frame, bbox)
            if features is not None:
                detection_features.append(features)
                valid_detections.append(det)
        
        if not valid_detections:
            # No valid detections in the current frame, update track last_seen_frame
            self._update_unseen_tracks(current_frame_idx)
            return [], []

        # Create cost matrix for matching: rows are new detections, columns are existing tracks
        cost_matrix = np.zeros((len(valid_detections), len(self.tracks)))
        track_ids_list = list(self.tracks.keys())

        for i, det_feat in enumerate(detection_features):
            for j, track_id in enumerate(track_ids_list):
                track_feat = self.tracks[track_id]['feature']
                # Cost is 1 - similarity (lower cost for higher similarity)
                cost_matrix[i, j] = 1 - self._calculate_cosine_similarity(det_feat, track_feat)

        # Use Hungarian algorithm for optimal assignment
        # row_ind: indices of new detections, col_ind: indices of existing tracks
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_detection_indices = set()
        matched_track_indices = set()

        # Process matched pairs
        for r, c in zip(row_ind, col_ind):
            similarity = 1 - cost_matrix[r, c]
            if similarity > self.threshold:
                # Match found: update existing track
                track_id = track_ids_list[c]
                matched_detection_indices.add(r)
                matched_track_indices.add(c)

                # Update track with new bbox and potentially blend features
                # For simplicity, we just update the bbox and feature
                self.tracks[track_id]['bbox'] = valid_detections[r]['bbox']
                self.tracks[track_id]['feature'] = detection_features[r] # Update with new feature
                self.tracks[track_id]['last_seen_frame'] = current_frame_idx
                
                current_bboxes.append(valid_detections[r]['bbox'])
                current_ids.append(track_id)
            
        # Process unmatched new detections (create new tracks)
        for i, det in enumerate(valid_detections):
            if i not in matched_detection_indices:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'bbox': det['bbox'],
                    'feature': detection_features[i],
                    'last_seen_frame': current_frame_idx
                }
                current_bboxes.append(det['bbox'])
                current_ids.append(new_id)

        # Update last_seen_frame for tracks that were not matched in this frame
        self._update_unseen_tracks(current_frame_idx, matched_track_indices, track_ids_list)
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_frame_idx)

        return current_bboxes, current_ids

    def _update_unseen_tracks(self, current_frame_idx, matched_track_indices=None, track_ids_list=None):
        """
        Updates the last_seen_frame for tracks that were not matched in the current frame.
        """
        # This function primarily ensures that if a track was NOT matched in the current frame,
        # its 'last_seen_frame' remains unchanged from the frame it was last seen.
        # The cleanup function then uses this 'last_seen_frame' to determine if it's too old.
        # No explicit action is needed here beyond what the matching loop already does
        # (i.e., only matched tracks have their 'last_seen_frame' updated to current_frame_idx).
        pass


    def _cleanup_old_tracks(self, current_frame_idx):
        """
        Removes tracks that haven't been seen for a specified number of frames.
        """
        tracks_to_delete = []
        for track_id, track_info in self.tracks.items():
            if (current_frame_idx - track_info['last_seen_frame']) > self.max_frames_unseen:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            # print(f"Track {track_id} terminated due to inactivity.")

