from __future__ import annotations

import json
import os
from pathlib import Path

import requests  # type: ignore
from pydantic import model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


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
]


def download_file(file_url: str, file_path: str):
    print(f'Downloading file {file_url} to {file_path}...')
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f'Model downloaded successfully to {file_path}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading model: {e}')


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Spoof Detection API'
    APP_ENV: str = 'production'
    CORS_ALLOW_ORIGINS: list[str] = []

    MODEL_PATH: str = str(BASE_DIR / 'spoofing_detection_api/models/model.pt')
    PARAMS_PATH: str = str(
        BASE_DIR / 'spoofing_detection_api/models/params.json')
    MODEL_THRESHOLD: float = 0.5
    API_V1_PREFIX: str = '/api/v1'
    MODEL_TARGET_SIZE: int = 320
    SPOOFING_MODEL_DOWNLOADS_URL_ENV: str = ''
    SPOOFING_PARAMS_DOWNLOAD_URL_ENV: str = ''

    @model_validator(mode='after')
    def load_model_params(self):
        try:
            if os.path.isfile(self.PARAMS_PATH) and os.path.isfile(self.MODEL_PATH):
                print('Model and params file found locally, loading params.')
            else:
                print('Params file not found at, downloading needed files.')
                download_file(
                    file_url=self.SPOOFING_MODEL_DOWNLOADS_URL_ENV,
                    file_path=self.MODEL_PATH,
                )
                download_file(
                    file_url=self.SPOOFING_PARAMS_DOWNLOAD_URL_ENV,
                    file_path=self.PARAMS_PATH,
                )

            with open(self.PARAMS_PATH) as f:
                params = json.load(f)
                self.MODEL_THRESHOLD = params.get(
                    'threshold',
                    self.MODEL_THRESHOLD,
                )
                self.MODEL_TARGET_SIZE = params.get(
                    'target_size',
                    self.MODEL_TARGET_SIZE,
                )

        except Exception as e:
            print(f'Error loading model parameters: {e}')
        # NOTE: add feature for deleting old files
        return self

    @model_validator(mode='after')
    def set_cors_origins(self):
        if self.APP_ENV == 'development':
            self.CORS_ALLOW_ORIGINS = allowed_origins_production + allowed_origins_development
        elif self.APP_ENV == 'production':
            self.CORS_ALLOW_ORIGINS = allowed_origins_production
        else:
            raise ValueError(
                'APP_ENV must be either "development" or "production".')
        return self

    class Config:
        env_file = str(BASE_DIR / '.env')


settings = Settings()

print(f"Looking for hahahahah .env at: {str(BASE_DIR / '.env')}")
