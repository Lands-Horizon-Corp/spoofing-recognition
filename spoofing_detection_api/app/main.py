from __future__ import annotations

import os
from typing import Union

from app.api.v1.routers import api_router
from app.core import utils
from app.core.config import settings
from app.core.security import limiter
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    # 1. Hide Swagger UI (/docs)
    docs_url='/docs' if settings.APP_ENV == 'development' else None,
    # 2. Hide ReDoc (/redoc)
    redoc_url='/redoc' if settings.APP_ENV == 'development' else None,
    # 3. (Optional) Hide the openapi.json schema file itself
    openapi_url='/openapi.json' if settings.APP_ENV == 'development' else None,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here (e.g., load model, initialize resources)
    print('Starting up the API...')

    if os.path.isfile(settings.PARAMS_PATH) and os.path.isfile(settings.MODEL_PATH):
        print('Model and params file found locally, loading params.')
    else:
        print('Params file not found at, downloading needed files.')
        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(settings.PARAMS_PATH), exist_ok=True)
        await utils.download_file(
            file_url=settings.SPOOFING_MODEL_DOWNLOADS_URL_ENV,
            file_path=settings.MODEL_PATH,
        )
        await utils.download_file(
            file_url=settings.SPOOFING_PARAMS_DOWNLOAD_URL_ENV,
            file_path=settings.PARAMS_PATH,
        )
    yield
    # Perform any shutdown tasks here (e.g., release resources)
    print('Shutting down the API...')


origin = settings.CORS_ALLOW_ORIGINS

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


app.include_router(api_router, prefix='/api/v1')


@app.get('/health', response_model=Union[dict, str])
async def health_check():
    """Health check endpoint to verify that the API is running."""
    return {'status': 'ok'}
