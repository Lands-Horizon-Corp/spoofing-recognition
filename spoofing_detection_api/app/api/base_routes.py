from __future__ import annotations

from app.api.v1.routers import v1_router
from robyn import SubRouter

base_route = SubRouter(__file__, prefix='')
base_route.include_router(v1_router)


@base_route.get('/')
async def root():
    """Root endpoint to verify that the API is running."""
    return {'message': 'Welcome to the Spoofing Detection API!'}


@base_route.get('/health')
async def health_check():
    """Health check endpoint to verify that the API is running."""
    return {'status': 'ok'}
