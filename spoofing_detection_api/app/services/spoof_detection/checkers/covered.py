from __future__ import annotations

import numpy as np
from app.core.utils import TBlendshapes
from app.core.utils import TFaceLandmarks

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh 468-point indices used for geometric checks
# ---------------------------------------------------------------------------

# Mouth Aspect Ratio (MAR) landmarks
_MOUTH_UPPER_INNER = 13   # upper inner lip mid-point
_MOUTH_LOWER_INNER = 14   # lower inner lip mid-point
_MOUTH_LEFT_CORNER = 61   # left commissure
_MOUTH_RIGHT_CORNER = 291  # right commissure

# Mouth visibility sentinel landmarks (must stay within image bounds)
_MOUTH_SENTINEL_INDICES = [13, 14, 61, 78, 291,
                           308, 0, 17, 37, 267, 84, 181, 91, 146,]

# Eye Aspect Ratio (EAR) landmarks
# Left eye:  outer=33, top=159, inner=133, bottom=145
# Right eye: outer=362, top=386, inner=263, bottom=374
_LEFT_EYE = (33, 159, 133, 145)
_RIGHT_EYE = (362, 386, 263, 374)

# Thresholds
_MAR_OPEN_THRESHOLD = 0.3   # MAR >= this → mouth is open
_EAR_CLOSED_THRESHOLD = 0.15  # EAR <= this → eye is closed


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def compute_mouth_aspect_ratio(pixel_points: np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) from pixel-space landmark coordinates.

    MAR = vertical_opening / horizontal_width

    A closed mouth typically yields MAR < 0.15; a clearly open mouth > 0.3.

    :param pixel_points: ndarray of shape (468, 2) with (x, y) pixel coords.
    :return: MAR value in [0, ∞).
    """
    upper = pixel_points[_MOUTH_UPPER_INNER]
    lower = pixel_points[_MOUTH_LOWER_INNER]
    left = pixel_points[_MOUTH_LEFT_CORNER]
    right = pixel_points[_MOUTH_RIGHT_CORNER]

    vertical = _dist(upper, lower)
    horizontal = _dist(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def compute_eye_aspect_ratio(pixel_points: np.ndarray, eye: str = 'left') -> float:
    """
    Compute Eye Aspect Ratio (EAR) from pixel-space landmark coordinates.

    EAR = vertical_opening / horizontal_width

    Open eyes typically yield EAR > 0.2; closed/occluded eyes ≤ 0.15.

    :param pixel_points: ndarray of shape (468, 2).
    :param eye: ``'left'`` or ``'right'``.
    :return: EAR value in [0, ∞).
    """
    outer, top, inner, bottom = _LEFT_EYE if eye == 'left' else _RIGHT_EYE
    vertical = _dist(pixel_points[top], pixel_points[bottom])
    horizontal = _dist(pixel_points[outer], pixel_points[inner])
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


# ---------------------------------------------------------------------------
# Checker functions
# ---------------------------------------------------------------------------

def is_eyes_covered(
    blendshapes: TBlendshapes,
    pixel_points: np.ndarray | None = None,
) -> bool:
    """
    Return True when either eye appears closed or occluded.

    Two independent signals are combined:

    * **Blendshapes** — ``eyeBlinkLeft / eyeBlinkRight`` score > 0.5 indicates
      the eyelid is largely closed (glasses occlusion scores low here, which is
      why the :class:`GlassChecker` handles that case separately).
    * **EAR** (optional) — if ``pixel_points`` are supplied, the geometric Eye
      Aspect Ratio is checked as a secondary confirmation.

    :param blendshapes: Blendshape score dict from MediaPipe FaceLandmarker.
    :param pixel_points: Optional (468, 2) pixel-coordinate array.
    :return: True if either eye is covered / closed.
    """
    left_blink = blendshapes.get('eyeBlinkLeft', 0.0)
    right_blink = blendshapes.get('eyeBlinkRight', 0.0)

    # Blendshape signal: both eyes closed simultaneously (blink or occlusion)
    if left_blink > 0.5 and right_blink > 0.5:
        return True

    # Geometric EAR signal (secondary)
    if pixel_points is not None and len(pixel_points) >= 468:
        ear_left = compute_eye_aspect_ratio(pixel_points, 'left')
        ear_right = compute_eye_aspect_ratio(pixel_points, 'right')
        if ear_left <= _EAR_CLOSED_THRESHOLD and ear_right <= _EAR_CLOSED_THRESHOLD:
            return True

    return False


def is_mouth_covered(
    face_landmarks: TFaceLandmarks,
    visibility_threshold: float = 0.6,
    min_visible_points: int = 4
) -> bool:
    """
    Return True if the mouth appears occluded (covered) based on landmark visibility.

    :param face_landmarks: List of NormalizedLandmark from FaceLandmarker.
    :param visibility_threshold: Minimum visibility score to consider a landmark "visible".
    :param min_visible_points: Required number of mouth landmarks with visibility >= threshold.
    :return: True if mouth is likely covered.
    """
    if not face_landmarks:
        return True  # No landmarks – definitely not visible

    # Count how many sentinel landmarks have good visibility
    visible_count = 0
    for idx in _MOUTH_SENTINEL_INDICES:  # or a dedicated set
        lm = face_landmarks[idx]
        if lm.visibility is not None and lm.visibility >= visibility_threshold:
            visible_count += 1
        print(f"Mouth landmark {idx} visibility: {lm.visibility}")
    print(f"Visible mouth landmarks: {
          visible_count}/{len(_MOUTH_SENTINEL_INDICES)}")
    # Mouth is covered if too few landmarks are clearly visible
    return visible_count < min_visible_points


def is_mouth_closed(
    blendshapes: TBlendshapes,
    pixel_points: np.ndarray | None = None,
    jaw_threshold: float = 0.6,
    mar_threshold: float = _MAR_OPEN_THRESHOLD,
) -> bool:
    """
    Return True when the mouth is closed (neutral / resting position).

    Two independent signals are used when available:

    * **jawOpen blendshape** — a score ≥ ``jaw_threshold`` means the jaw is
      detectably open.
    * **MAR** — if ``pixel_points`` are supplied, Mouth Aspect Ratio is
      computed; MAR ≥ ``mar_threshold`` confirms the mouth is open.

    The mouth is considered *open* (returns False) if **either** signal fires,
    giving a conservative check suitable for liveness pre-screening.

    :param blendshapes: Blendshape score dict from MediaPipe FaceLandmarker.
    :param pixel_points: Optional (468, 2) pixel-coordinate array.
    :param jaw_threshold: jawOpen score above which the mouth is open (0–1).
    :param mar_threshold: MAR above which the mouth is open (dimensionless).
    :return: True if mouth appears closed.
    """
    if not blendshapes:
        return False

    # Blendshape signal
    if blendshapes.get('jawOpen', 0.0) >= jaw_threshold:
        return False

    # Geometric MAR signal (secondary)
    if pixel_points is not None and len(pixel_points) >= 292:
        if compute_mouth_aspect_ratio(pixel_points) >= mar_threshold:
            return False

    return True
