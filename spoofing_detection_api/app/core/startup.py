from __future__ import annotations

import os

from app.core import utils
from app.core.config import settings


async def download_model():
    """Downloads the model and params files if they do not exist locally."""
    if os.path.isfile(settings.PARAMS_PATH) and os.path.isfile(settings.MODEL_PATH):
        print('Model and params file found locally, loading params.')
        assert os.path.isfile(settings.PARAMS_PATH), 'Params file is missing'
        assert os.path.isfile(settings.MODEL_PATH), 'Model file is missing'
        assert os.path.isfile(
            settings.FACE_DETECTOR_MODEL_PATH), 'Face detector model file is missing'
        assert os.path.isfile(
            settings.GLASS_DETECTOR_MODEL_PATH), 'Glass detector model file is missing'
        assert os.path.isfile(
            settings.FACE_LANDMARKS_MODEL_PATH), 'Face landmarks model missing'

    else:
        print('Params file not found at, downloading needed files.')

        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(settings.PARAMS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(
            settings.FACE_DETECTOR_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(
            settings.GLASS_DETECTOR_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(
            settings.FACE_LANDMARKS_MODEL_PATH), exist_ok=True)

        await utils.download_file(
            file_url=settings.SPOOFING_MODEL_DOWNLOADS_URL_ENV,
            file_path=settings.MODEL_PATH,
        )

        await utils.download_file(
            file_url=settings.SPOOFING_PARAMS_DOWNLOAD_URL_ENV,
            file_path=settings.PARAMS_PATH,
        )

        await utils.download_file(
            file_url=settings.SPOOFING_FACE_DETECTOR_DOWNLOAD_URL_ENV,
            file_path=settings.FACE_DETECTOR_MODEL_PATH,
        )
        await utils.download_file(
            file_url=settings.GLASS_DETECTOR_MODEL_DOWNLOAD_URL_ENV,
            file_path=settings.GLASS_DETECTOR_MODEL_PATH,
        )
        await utils.download_file(
            file_url=settings.FACE_LANDMARKS_MODEL_DOWNLOAD_URL_ENV,
            file_path=settings.FACE_LANDMARKS_MODEL_PATH,
        )
        assert os.path.isfile(
            settings.MODEL_PATH), 'Model file was not downloaded successfully'
        assert os.path.isfile(
            settings.PARAMS_PATH), 'Params file was not downloaded successfully'
        assert os.path.isfile(
            settings.FACE_DETECTOR_MODEL_PATH), 'Face detector file not downloaded successfully'
        assert os.path.isfile(
            settings.GLASS_DETECTOR_MODEL_PATH), 'Glass detector file not downloaded successfully'
        assert os.path.isfile(
            settings.FACE_LANDMARKS_MODEL_PATH), 'Face landmarks model file not downloaded'


# async def load_model() -> onnxruntime.InferenceSession:
#     """Loads the ONNX model into memory."""
#     return spoof_detector.load_model()
