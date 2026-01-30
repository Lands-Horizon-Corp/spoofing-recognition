from __future__ import annotations

import json
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = 'Spoof Detection API'
    MODEL_PATH: str = str(BASE_DIR / 'models/model.pt')
    PARAMS_PATH: str = str(BASE_DIR / 'models/params.json')
    MODEL_THRESHOLD: float = 0.5
    API_V1_PREFIX: str = '/api/v1'
    MODEL_TARGET_SIZE: int = 320
    IS_INDEVELOPMENT: bool = False
    CORS_ALLOW_ORIGINS: str = '*'
    PORT: int = 8001

    @property
    def cors_origins(self) -> list[str]:
        if self.CORS_ALLOW_ORIGINS == '*':
            return ['*']
        return self.CORS_ALLOW_ORIGINS.split(',')

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

    class Config:
        env_file = str(BASE_DIR / '.env')


settings = Settings()
