from __future__ import annotations

import filetype
from app.services.detection_service import predict_spoof
from robyn import Request
from robyn import Response
from robyn import SubRouter

router = SubRouter(__file__, prefix='/api/v1/spoof')
print('Spoof Detection Routes Loaded')


def is_image_file(file: bytes) -> bool:
    """Check if the uploaded file is an image based on its content type."""

    file_Info = filetype.guess(file)
    return file_Info is not None and file_Info.mime.startswith('image/')


@router.get('/ping')
async def ping():
    return {'ping': 'pong'}


@router.post('/detect')
async def detect_spoof(request: Request):
    """Endpoint to detect spoofing in an uploaded image"""
    files = request.files
    file_names = files.keys()
    print({'file_names': list(file_names)})
    first_key = list(file_names)[0]
    img = files[first_key]
    # Debug: Print available files)

    if not is_image_file(img):
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description='{"error": "File must be an image"}'
        )

    try:
        result = await predict_spoof(img)
    except ValueError as e:
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description=f'{{"error": "{str(e)}"}}'
        )

    return result


@router.post('/detect/verbose')
async def detect_spoof_verbose(request: Request):
    """Endpoint to detect spoofing and return 204 if live, 401 if spoof"""
    # Try to get the file with 'file' key
    file = request.files.get('file')

    # If not found, try the first available file
    if not file and request.files:
        first_key = list(request.files.keys())[0]
        file = request.files[first_key]

    if not file:
        available_keys = list(request.files.keys()) if request.files else []
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description=f'{{"error": "No file uploaded", "available_keys": {
                available_keys}}}'
        )

    if not is_image_file(file):
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description='{"error": "File must be an image"}'
        )

    try:
        result = await predict_spoof(file)
    except ValueError as e:
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description=f'{{"error": "{str(e)}"}}'
        )

    if result['is_spoof']:
        return Response(
            status_code=401,
            headers={'Content-Type': 'application/json'},
            description='{"status": "Spoof detected"}'
        )

    return Response(
        status_code=204,
        headers={'Content-Type': 'application/json'},
        description='{"status": "Live detected"}'
    )
