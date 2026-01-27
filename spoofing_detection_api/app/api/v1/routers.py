from fastapi import APIRouter
from app.api.v1.endpoints import spoof_detection

api_router = APIRouter()

# URL becomes /api/v1/spoof/detect
api_router.include_router(spoof_detection.router, prefix="/spoof", tags=["Spoofing"])
