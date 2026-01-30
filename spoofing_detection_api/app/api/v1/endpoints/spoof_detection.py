from __future__ import annotations

from app.schemas.detection import DetectionResult
from app.services.detection_service import predict_spoof
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Response
from fastapi import status
from fastapi import UploadFile

router = APIRouter()


@router.post('/detect', response_model=DetectionResult)
async def detect_spoof(file: UploadFile = File(...)):
    """Endpoint to detect spoofing in an uploaded image"""

    if file.content_type is None:
        raise HTTPException(400, 'No file uploaded')

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')
    try:
        result = await predict_spoof(file)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return result


# /verbose


@router.post('/detect/verbose', status_code=status.HTTP_204_NO_CONTENT)
async def detect_spoof_verbose(response: Response, file: UploadFile = File(...)):
    """Endpoint to detect spoofing in an uploaded image"""

    if file.content_type is None:
        raise HTTPException(400, 'No file uploaded')

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, 'File must be an image')
    try:
        result = await predict_spoof(file)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 401 return a spoof no json
    if result['is_spoof']:
        raise HTTPException(400, 'Spoof detected')

    # 204 return a live no json
    response.status_code = status.HTTP_204_NO_CONTENT
    return None
