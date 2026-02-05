from __future__ import annotations

import json
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Spoof Detection API'
    APP_ENV: str = 'production'
    CORS_ALLOW_ORIGINS: list[str] = []

    MODEL_PATH: str = str(BASE_DIR / 'models/model.pt')
    PARAMS_PATH: str = str(BASE_DIR / 'models/params.json')
    MODEL_THRESHOLD: float = 0.5
    API_V1_PREFIX: str = '/api/v1'
    MODEL_TARGET_SIZE: int = 320

    @model_validator(mode='after')
    def load_model_params(self):
        try:
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
        except FileNotFoundError:
            print('Params file not found, using default parameters.')
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

print(f"Looking for .env at: {str(BASE_DIR / '.env')}")
