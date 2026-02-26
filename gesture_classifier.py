"""
Gesture classification module for recognizing different hand gestures.
"""
import numpy as np
from typing import List, Optional, Tuple


class GestureClassifier:
    """Classify hand gestures based on finger positions and hand landmarks."""
    
    def __init__(self):
        """Initialize the gesture classifier."""
        self.gesture_names = [
            "Unknown",
            "Fist",
            "Open Hand",
            "Thumbs Up",
            "Thumbs Down",
            "Peace Sign",
            "OK Sign",
            "Pointing",
            "Three",
            "Four",
            "Five"
        ]
    
    def classify_gesture(self, landmarks: Optional[np.ndarray], 
                        finger_count: int = 0,
                        hand_label: Optional[str] = None) -> Tuple[int, str]:
        """
        Classify hand gesture based on landmarks and finger count.
        
        Args:
            landmarks: Array of hand landmarks (21, 2) or None.
            finger_count: Number of extended fingers.
            hand_label: "Left" or "Right" hand label.
            
        Returns:
            Tuple of (gesture_id, gesture_name).
        """
        if landmarks is None:
            return 0, self.gesture_names[0]
        
        # Normalize landmarks relative to wrist (landmark 0)
        normalized_landmarks = landmarks - landmarks[0]
        
        # Get gesture based on finger count and landmark analysis
        gesture_id = self._analyze_gesture(normalized_landmarks, finger_count)
        
        return gesture_id, self.gesture_names[gesture_id]
    
    def _analyze_gesture(self, landmarks: np.ndarray, finger_count: int) -> int:
        """
        Analyze landmarks to determine gesture type.
        
        Args:
            landmarks: Normalized landmark array (21, 2).
            finger_count: Number of extended fingers.
            
        Returns:
            Gesture ID.
        """
        # Tip IDs: thumb=4, index=8, middle=12, ring=16, pinky=20
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # MCP (knuckle) positions for comparison
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Check individual finger positions
        index_up = index_tip[1] < index_mcp[1]
        middle_up = middle_tip[1] < middle_mcp[1]
        ring_up = ring_tip[1] < ring_mcp[1]
        pinky_up = pinky_tip[1] < pinky_mcp[1]
        
        # Thumb position (check x coordinate for thumb)
        thumb_extended = abs(thumb_tip[0]) > abs(landmarks[3][0])
        
        # Gesture classification logic
        if finger_count == 0:
            return 1  # Fist
        
        elif finger_count == 5:
            return 2  # Open Hand
        
        elif finger_count == 1:
            if thumb_extended and not index_up and not middle_up:
                # Check if thumb is up or down
                thumb_y_diff = thumb_tip[1] - landmarks[0][1]
                if thumb_y_diff < -0.05:
                    return 3  # Thumbs Up
                elif thumb_y_diff > 0.05:
                    return 4  # Thumbs Down
            elif index_up and not thumb_extended:
                return 7  # Pointing
        
        elif finger_count == 2:
            if index_up and middle_up and not ring_up and not pinky_up:
                return 5  # Peace Sign
            elif thumb_extended and index_up:
                # Check for OK sign (thumb and index form circle)
                thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
                if thumb_index_distance < 0.03:
                    return 6  # OK Sign
        
        elif finger_count == 3:
            if index_up and middle_up and ring_up:
                return 8  # Three
        
        elif finger_count == 4:
            if index_up and middle_up and ring_up and pinky_up:
                return 9  # Four
        
        # Default fallback
        return 0  # Unknown


