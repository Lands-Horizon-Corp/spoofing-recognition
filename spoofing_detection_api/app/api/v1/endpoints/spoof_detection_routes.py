from __future__ import annotations

import json

import filetype
from app.core.constants.http_status import HTTPStatus
from app.core.constants.http_status import SpoofVerboseHTTPStatus
from app.core.middleware import header_builder
from app.core.middleware import resolve_origin
from app.services.detection_service import predict_spoof
from robyn import Request
from robyn import Response
from robyn import SubRouter
from spoofdet.verify_memory import print_memory_usage

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
        print_memory_usage('Error during prediction')
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description='{"error": "File must be an image"}'
        )

    try:
        result = await predict_spoof(img)
    except ValueError as e:
        print_memory_usage('Error during prediction')
        return Response(
            status_code=400,
            headers={'Content-Type': 'application/json'},
            description=f'{{"error": "{str(e)}"}}'
        )
    print_memory_usage('Prediction completed')
    return result


@router.post('/detect/verbose')
async def detect_spoof_verbose(request: Request):
    """Endpoint to detect spoofing and return 204 if live, 401 if spoof"""
    # Try to get the file with 'file' key
    files = request.files
    file_names = list(files.keys())
    print('POST: /detect/verbose', {'file_names': file_names})

    origin = request.headers.get('origin')
    allowed_origin = resolve_origin(origin)
    headers = header_builder(allowed_origin)
    print(f"Request origin: {origin}")

    if not file_names:
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description='{"error": "No file uploaded"}'
        )

    first_key = file_names[0]
    img = files[first_key]

    if not img:
        available_keys = list(request.files.keys()) if request.files else []
        error = 'No file uploaded'
        descriptions = json.dumps({'error': error, 'details': available_keys})
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description=descriptions
        )

    if not is_image_file(img):
        return Response(
            status_code=HTTPStatus.BAD_REQUEST.value,
            headers=headers,
            description='{"error": "File must be an image"}'
        )

    try:
        result = await predict_spoof(img)
    except ValueError as e:
        if 'No faces detected' in str(e):
            status_code = SpoofVerboseHTTPStatus.NO_FACE.value
        elif 'Multiple faces detected' in str(e):
            status_code = SpoofVerboseHTTPStatus.MULTIPLE_FACES.value
        elif 'face forward properly' in str(e):
            status_code = SpoofVerboseHTTPStatus.NOT_FRONTAL.value
        else:
            status_code = HTTPStatus.BAD_REQUEST.value
        return Response(
            status_code=status_code,
            headers=headers,
            description=f'{{"error": "{str(e)}"}}'
        )

    if result['is_spoof']:
        return Response(
            status_code=SpoofVerboseHTTPStatus.SPOOF_DETECTED.value,
            headers=headers,
            description='spoof detected'
        )

    return Response(
        status_code=HTTPStatus.OK_NO_CONTENT.value,
        headers=headers,
        description=''
    )
