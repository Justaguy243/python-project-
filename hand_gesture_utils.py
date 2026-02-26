"""
Utility functions for hand gesture detection and processing.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class HandDetector:
    """Hand detection and landmark extraction using MediaPipe."""
    
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the hand detector.
        
        Args:
            static_image_mode: If True, detection runs on every input image.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and draw landmarks if requested.
        
        Args:
            img: Input image (BGR format).
            draw: Whether to draw hand landmarks on the image.
            
        Returns:
            Tuple of (image with/without drawings, hand landmarks).
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        
        return img, self.results
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Extract landmark positions from detected hands.
        
        Args:
            img: Input image.
            hand_no: Which hand to process (0 or 1 for multi-hand).
            draw: Whether to draw circles on landmarks.
            
        Returns:
            List of landmark positions [(id, x, y), ...].
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list
    
    def get_landmarks_array(self, img, hand_no=0):
        """
        Get normalized landmark coordinates as a numpy array.
        
        Args:
            img: Input image.
            hand_no: Which hand to process.
            
        Returns:
            Numpy array of shape (21, 2) with normalized coordinates, or None.
        """
        if not self.results.multi_hand_landmarks:
            return None
        
        if hand_no >= len(self.results.multi_hand_landmarks):
            return None
        
        hand = self.results.multi_hand_landmarks[hand_no]
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y])
        
        return np.array(landmarks)
    
    def count_fingers(self, landmarks, hand_label=None):
        """
        Count the number of extended fingers.
        
        Args:
            landmarks: List of landmark positions [(id, x, y), ...].
            hand_label: "Left" or "Right" hand label for better thumb detection.
            
        Returns:
            Number of extended fingers (0-5).
        """
        if len(landmarks) == 0:
            return 0
        
        # Finger tip IDs: thumb, index, middle, ring, pinky
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        
        # Thumb (special case - direction depends on hand)
        # For right hand (in mirrored view): thumb extended if tip x > IP joint x
        # For left hand (in mirrored view): thumb extended if tip x < IP joint x
        thumb_tip_id = tip_ids[0]
        thumb_ip_id = thumb_tip_id - 1  # IP joint
        
        if thumb_tip_id < len(landmarks) and thumb_ip_id < len(landmarks):
            if hand_label == "Left":
                # Left hand: thumb extends to the left (smaller x in mirrored view)
                thumb_up = landmarks[thumb_tip_id][1] < landmarks[thumb_ip_id][1]
            else:
                # Right hand (default): thumb extends to the right (larger x in mirrored view)
                thumb_up = landmarks[thumb_tip_id][1] > landmarks[thumb_ip_id][1]
            
            # Alternative: check if thumb tip is away from hand center
            # This works better when hand is rotated
            wrist_x = landmarks[0][1] if len(landmarks) > 0 else 0
            thumb_tip_x = landmarks[thumb_tip_id][1]
            thumb_ip_x = landmarks[thumb_ip_id][1]
            
            # Thumb is up if tip is further from wrist than IP joint
            thumb_extended = abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x) * 1.2
            
            fingers.append(1 if thumb_extended else 0)
        else:
            fingers.append(0)
        
        # Other fingers (compare y coordinates - tips should be above MCP joints)
        for id in range(1, 5):
            tip_id = tip_ids[id]
            pip_id = tip_id - 2  # PIP joint (more reliable than MCP)
            
            if tip_id < len(landmarks) and pip_id < len(landmarks):
                # Finger is extended if tip is above PIP joint
                if landmarks[tip_id][2] < landmarks[pip_id][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def get_hand_label(self, hand_no=0):
        """
        Get the label (Left/Right) of the detected hand.
        
        Args:
            hand_no: Which hand to process.
            
        Returns:
            String label ("Left" or "Right"), or None.
        """
        if not self.results.multi_handedness:
            return None
        
        if hand_no >= len(self.results.multi_handedness):
            return None
        
        return self.results.multi_handedness[hand_no].classification[0].label

