from __future__ import annotations

from app.core.config import settings
from app.core.utils import DownloadFile


async def download_models():
    download_spoof_model = DownloadFile(
        file_url=settings.SPOOFING_MODEL_DOWNLOADS_URL_ENV,
        file_path=settings.MODEL_PATH,
    )
    download_spoof_params = DownloadFile(
        file_url=settings.SPOOFING_PARAMS_DOWNLOAD_URL_ENV,
        file_path=settings.PARAMS_PATH,
    )
    download_face_detector_model = DownloadFile(
        file_url=settings.SPOOFING_FACE_DETECTOR_DOWNLOAD_URL_ENV,
        file_path=settings.FACE_DETECTOR_MODEL_PATH,
    )
    download_glass_detector_model = DownloadFile(
        file_url=settings.GLASS_DETECTOR_MODEL_DOWNLOAD_URL_ENV,
        file_path=settings.GLASS_DETECTOR_MODEL_PATH,
    )
    download_face_landmarks_model = DownloadFile(
        file_url=settings.FACE_LANDMARKS_MODEL_DOWNLOAD_URL_ENV,
        file_path=settings.FACE_LANDMARKS_MODEL_PATH,
    )

    downloads = [download_spoof_model, download_spoof_params, download_face_detector_model,
                 download_glass_detector_model, download_face_landmarks_model]

    for download in downloads:
        if not download.check_file_exists():
            await download.execute()
        else:
            print(
                f'File {download.file_path} already exists, skipping download.')
