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
    min_inbound_points: int = 10,
) -> bool:
    """
    Return True if the mouth appears occluded or out-of-frame.

    MediaPipe FaceLandmarker does not populate the ``visibility`` attribute
    (it is always ``None``).  Instead we rely on whether the normalized (x, y)
    coordinates of sentinel mouth landmarks fall within the [0, 1] image frame.
    A landmark that lands outside that range is off-screen, meaning the mouth
    is at least partially cropped or not reliably detected.

    :param face_landmarks: List of NormalizedLandmark from FaceLandmarker.
    :param min_inbound_points: Minimum number of mouth landmarks that must be
        within [0, 1] bounds to consider the mouth visible.
    :return: True if mouth is likely covered / out of frame.
    """
    if not face_landmarks:
        return True  # no landmarks — face/mouth not visible

    inbound = sum(
        1
        for idx in _MOUTH_SENTINEL_INDICES
        if 0.0 <= face_landmarks[idx].x <= 1.0
        and 0.0 <= face_landmarks[idx].y <= 1.0
    )
    print(
        f'Mouth visibility check: {inbound} in-bounds points (threshold: {min_inbound_points})')
    return inbound < min_inbound_points


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
