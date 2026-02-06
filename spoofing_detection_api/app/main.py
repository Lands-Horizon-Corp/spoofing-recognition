from __future__ import annotations

from typing import Union

from app.api.v1.routers import api_router
from app.core.config import settings
from app.core.security import limiter
from fastapi import FastAPI
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
