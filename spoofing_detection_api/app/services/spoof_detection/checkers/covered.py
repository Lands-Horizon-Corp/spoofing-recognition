from __future__ import annotations

from app.core.utils import TBlendshapes
from app.core.utils import TFaceLandmarks


def is_eyes_covered(blendshapes: TBlendshapes) -> bool:
    left_blink_score = blendshapes.get('eyeBlinkLeft', 0.0)
    right_blink_score = blendshapes.get('eyeBlinkRight', 0.0)
    is_eyes_covered = (
        left_blink_score > 0.5
        and right_blink_score > 0.5
    )
    return is_eyes_covered


def is_mouth_covered(face_landmarks: TFaceLandmarks) -> bool:
    if not face_landmarks:
        return True  # no landmarks detected — face/mouth not visible

    mouth_indices = [13, 14, 78, 308]
    for idx in mouth_indices:
        lm = face_landmarks[idx]
        # FaceLandmarker normalizes to [0,1], but landmarks near the edge
        # of a cropped face image can still fall slightly outside bounds
        if lm.x < 0.02 or lm.x > 0.98 or lm.y < 0.02 or lm.y > 0.98:
            return True
    return False


def is_mouth_closed(blendshapes: TBlendshapes, threshold: float = 0.6) -> bool:
    if not blendshapes:
        return False
    jaw_open = blendshapes.get('jawOpen', 0.0)
    return jaw_open < threshold
