from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.detection import DetectionResult
from app.services.detection_service import predict_spoof
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image
import io
import numpy as np

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect_spoof(file: UploadFile = File(...)):
    """Endpoint to detect spoofing in an uploaded image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        result = await predict_spoof(file)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return result
