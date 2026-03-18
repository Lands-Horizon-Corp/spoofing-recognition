from __future__ import annotations

import asyncio

import filetype
from app.core.constants.http_status import HTTPStatus
from app.core.constants.http_status import SpoofVerboseHTTPStatus
from app.core.middleware import header_builder
from app.core.middleware import resolve_origin
from app.schemas.detection import DetectionResult
from app.services.spoof_detection.detect import detect_spoof_service
from robyn import Request
from robyn import Response
from robyn import SubRouter
from spoofdet.utils.verify_memory import print_memory_usage

router = SubRouter(__file__, prefix='/api/v1/spoof')
print('Spoof Detection Routes Loaded')


def is_image_file(file: bytes) -> bool:
    """Check if the uploaded file is an image based on its content type."""

    file_Info = filetype.guess(file)
    return file_Info is not None and file_Info.mime.startswith('image/')


def get_image(request: Request) -> bytes | None:
    """Extract the image file from the request."""
    files = request.files
    if not files:
        return None
    file_names = list(files.keys())
    if not file_names:
        return None
    first_key = file_names[0]
    if not is_image_file(files[first_key]):
        return None
    print(f"Received file: {first_key}, size: {len(files[first_key])} bytes")
    return files[first_key]


@router.get('/ping')
async def ping():
    return {'ping': 'pong'}


@router.post('/detect')
async def detect_spoof(request: Request):
    """Endpoint to detect spoofing in an uploaded image"""
    # Debug: Print available files)
    origin = request.headers.get('origin')
    cors_req = request.headers.get('access-control-request-method')
    allowed_origin = resolve_origin(origin)
    headers = header_builder(allowed_origin, cors_req)
    img = get_image(request)
    if img is None:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description='{"code": "ERR_NO_FILE_UPLOADED"}'
        )

    try:
        result: DetectionResult = detect_spoof_service(img)

    except ValueError as e:
        print_memory_usage('Error during prediction')
        print(f"Error: {str(e)}")
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description=f'{{"code": "{str(e)}"}}'
        )

    if result.is_spoof:
        print_memory_usage('Spoof detected')
        return Response(
            status_code=HTTPStatus.UNAUTHORIZED.value,
            headers=headers,
            description='{"code": "SPOOF_DETECTED"}'
        )
    print_memory_usage('Prediction completed')
    return Response(
        status_code=HTTPStatus.OK.value,
        headers=headers,
        description=str(result)
    )


@router.post('/detect/verbose')
async def detect_spoof_verbose(request: Request):
    """Endpoint to detect spoofing and return 204 if live, 401 if spoof, and 400 for errors"""

    img = get_image(request)
    origin = request.headers.get('origin')
    cors_req = request.headers.get('access-control-request-method')
    allowed_origin = resolve_origin(origin)
    headers = header_builder(allowed_origin, cors_req)
    print(f"Request origin: {origin}")

    if not img:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description='{"code": "ERR_NO_FILE_UPLOADED"}'
        )

    try:
        result = await asyncio.to_thread(detect_spoof_service, img)
    except ValueError as e:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description=f'{{"code": "{str(e)}"}}'
        )

    if result.is_spoof:
        return Response(
            status_code=SpoofVerboseHTTPStatus.SPOOF_DETECTED.value,
            headers=headers,
            description='{"code": "SPOOF_DETECTED"}'
        )

    return Response(
        status_code=HTTPStatus.OK_NO_CONTENT.value,
        headers=headers,
        description=''
    )
