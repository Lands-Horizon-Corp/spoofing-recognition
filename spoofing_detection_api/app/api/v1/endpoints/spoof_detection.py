from __future__ import annotations

from app.core.security import limiter
from app.schemas.detection import DetectionResult
from app.services.detection_service import predict_spoof
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi import UploadFile

router = APIRouter()


@router.post('/detect', response_model=DetectionResult)
@limiter.limit('5/minute')
async def detect_spoof(request: Request, file: UploadFile = File(...)):
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
@limiter.limit('5/minute')
async def detect_spoof_verbose(request: Request, response: Response, file: UploadFile = File(...)):
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
        raise HTTPException(401, 'Spoof detected')
    if result['live_confidence'] < 0.90:
        raise HTTPException(403, 'Low confidence live detected')
    if result['live_confidence'] > 0.90:
        response.status_code = status.HTTP_204_NO_CONTENT

    # 204 return a live no json
    return None
