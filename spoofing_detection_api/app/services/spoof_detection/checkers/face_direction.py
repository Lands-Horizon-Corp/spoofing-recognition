from __future__ import annotations

import enum
import logging

from app.core.utils import TPose
from PIL import Image

logger = logging.getLogger(__name__)

# Thresholds (degrees) — tune per application requirements
_YAW_THRESHOLD = 15.0    # left / right rotation
_PITCH_THRESHOLD = 15.0  # up / down tilt
_ROLL_THRESHOLD = 10.0   # in-plane head tilt


class FaceDirectionChecker:
    """
    MediaPipe-based forward-facing checker.

    Call :meth:`is_facing_forward` with a PIL image; it returns True only
    when the face satisfies all three pose thresholds.  The full pipeline in
    ``detect.py`` uses :func:`check_face_direction` directly on the pose dict
    that is already extracted by ``mp_utils``, so this class is provided as a
    standalone convenience wrapper.
    """

    def is_facing_forward(self, image: Image.Image) -> bool:
        """
        Return True when the face is forward-facing (yaw/pitch/roll within
        thresholds).  Requires ``mp_utils`` to be loaded.

        :param image: RGB PIL image.
        :return: True if forward-facing.
        """
        from app.core.utils import mp_utils

        try:
            _, _, _, pose = mp_utils.extract_landmarks(image)
        except ValueError:
            return False

        return check_face_direction(pose) == FaceDirectionEnum.FRONTAL


face_direction_checker = FaceDirectionChecker()


class FaceDirectionEnum(enum.Enum):
    FRONTAL = 'FRONTAL'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    UP = 'UP'
    DOWN = 'DOWN'
    TILTED = 'TILTED'  # in-plane roll exceeds threshold


def check_face_direction(pos: TPose) -> FaceDirectionEnum:
    """
    Classify head orientation from a pose dict containing ``pitch``, ``yaw``,
    and ``roll`` (all in degrees, produced by :meth:`MediaPipeUtils.get_head_pose`).

    Checks are applied in priority order:

    1. Roll (in-plane tilt) — returns ``TILTED``
    2. Yaw (left / right)  — returns ``LEFT`` or ``RIGHT``
    3. Pitch (up / down)   — returns ``UP`` or ``DOWN``
    4. Otherwise           — returns ``FRONTAL``
    """
    pitch = pos.get('pitch')
    yaw = pos.get('yaw')
    roll = pos.get('roll')

    assert pitch is not None, 'Pitch value is missing in pose data'
    assert yaw is not None, 'Yaw value is missing in pose data'
    assert roll is not None, 'Roll value is missing in pose data'

    logger.debug('Pose — pitch: %.1f°  yaw: %.1f°  roll: %.1f°',
                 pitch, yaw, roll)

    if abs(roll) >= _ROLL_THRESHOLD:
        return FaceDirectionEnum.TILTED

    if yaw < -_YAW_THRESHOLD:
        return FaceDirectionEnum.LEFT
    elif yaw > _YAW_THRESHOLD:
        return FaceDirectionEnum.RIGHT

    if pitch < -_PITCH_THRESHOLD:
        return FaceDirectionEnum.DOWN
    elif pitch > _PITCH_THRESHOLD:
        return FaceDirectionEnum.UP

    return FaceDirectionEnum.FRONTAL
