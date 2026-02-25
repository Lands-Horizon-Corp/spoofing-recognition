from __future__ import annotations

import json
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


allowed_headers = [
    'Content-Type',
    'Accept',
    'Authorization',
    'Location',
    'X-Organization-Id',
    'X-User-Agent',
    'X-Device-Type',
    'X-CSRF-Token',
]


allowed_origins_production = [
    'https://ecoop-suite.netlify.app',
    'https://ecoop-suite.com',
    'https://www.ecoop-suite.com',

    'https://development.ecoop-suite.com',
    'https://www.development.ecoop-suite.com',
    'https://staging.ecoop-suite.com',
    'https://www.staging.ecoop-suite.com',

    'https://cooperatives-development.fly.dev',
    'https://cooperatives-staging.fly.dev',
    'https://cooperatives-production.fly.dev',

    'https://cooperatives-development-production-0fc5.up.railway.app',
    'https://e-coop-server-development.up.railway.app',
    'https://e-coop-server-production.up.railway.app',
    'https://e-coop-server-staging.up.railway.app',

    'https://e-coop-client-development.up.railway.app',
    'https://e-coop-client-production.up.railway.app',
    'https://e-coop-client-staging.up.railway.app',

    'https://spoofing-recognition-development.up.railway.app',
]

allowed_origins_development = [
    'http://localhost:8000',
    'http://localhost:8001',
    'http://localhost:3000',
    'http://localhost:3001',
    'http://localhost:3002',
    'http://localhost:3003',
    'http://localhost:4173',
    'http://localhost:4174',
    # /default/post_api_v1_spoof_detect
]


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Spoof Detection API'
    APP_ENV: str = 'production'
    CORS_ALLOW_ORIGINS: list[str] = []
    CORS_ALLOW_HEADERS: list[str] = allowed_headers
    MODEL_PATH: str = str(
        BASE_DIR / 'spoofing_detection_api/models/model.onnx')

    PARAMS_PATH: str = str(
        BASE_DIR / 'spoofing_detection_api/models/params.json')
    FACE_DETECTOR_MODEL_PATH: str = str(
        BASE_DIR / 'spoofing_detection_api/models/version-RFB-320_without_postprocessing.onnx')
    API_V1_PREFIX: str = '/api/v1'
    SPOOFING_MODEL_DOWNLOADS_URL_ENV: str = ''
    SPOOFING_PARAMS_DOWNLOAD_URL_ENV: str = ''
    SPOOFING_FACE_DETECTOR_DOWNLOAD_URL_ENV: str = ''
    THRESHOLD: float = 0.5
    OPENAPI_PATH: str = str(BASE_DIR / 'spoofing_detection_api/openapi.json')

    @model_validator(mode='after')
    def set_cors_origins(self):
        print(f'APP_ENV: {self.APP_ENV}')
        if self.APP_ENV == 'development':
            self.CORS_ALLOW_ORIGINS = allowed_origins_development + allowed_origins_production
        elif self.APP_ENV == 'production':
            self.CORS_ALLOW_ORIGINS = allowed_origins_production
        elif self.APP_ENV == 'staging':
            self.CORS_ALLOW_ORIGINS = allowed_origins_development + allowed_origins_production
        else:
            raise ValueError(
                'APP_ENV must be either "development", "production", or "staging".')
        return self

    class Config:
        env_file = str(BASE_DIR / '.env')


print(str(BASE_DIR / '.env'))

settings = Settings()


class ModelConfig(BaseSettings):
    THRESHOLD: float = 0.5
    TARGET_SIZE: int = 320

    def load_model_params(self, path: str = settings.PARAMS_PATH):
        try:
            with open(path) as f:
                params = json.load(f)
                self.THRESHOLD = params.get(
                    'threshold',
                    self.THRESHOLD,
                )
                self.TARGET_SIZE = params.get(
                    'target_size',
                    self.TARGET_SIZE,
                )

        except Exception as e:
            print(f'Error loading model parameters: {e}')
        # NOTE: add feature for deleting old files
        return self


model_config = ModelConfig()
