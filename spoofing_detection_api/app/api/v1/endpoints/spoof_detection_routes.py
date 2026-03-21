from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse

import filetype
from app.core.constants.http_status import HTTPStatus
from app.core.constants.http_status import SpoofVerboseHTTPStatus
from app.core.middleware import header_builder
from app.core.middleware import resolve_origin
from app.schemas.detection import DetectionResult
from app.services.spoof_detection.detect import detect_spoof_service
from pyinstrument import Profiler as PyInstrumentProfiler
from robyn import Request
from robyn import Response
from robyn import SubRouter
from spoofdet.utils.verify_memory import print_memory_usage

router = SubRouter(__file__, prefix='/api/v1/spoof')
print('Spoof Detection Routes Loaded')
LATEST_PROFILE_PATH = Path('./profile_latest.html')
PROFILE_ENABLED_HEADER = 'x-profile-enabled'


def _to_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _query_value(req: Request, key: str) -> object | None:
    query_params = getattr(req, 'query_params', None)
    if query_params is not None:
        try:
            query_dict = query_params.to_dict()
            if key in query_dict:
                return query_dict.get(key)
        except Exception:
            pass

        try:
            return query_params.get(key)
        except Exception:
            pass

    try:
        raw_url = str(getattr(req, 'url', ''))
        parsed_qs = parse_qs(urlparse(raw_url).query)
        values = parsed_qs.get(key)
        return values[0] if values else None
    except Exception:
        return None


def _is_profile_enabled(request: Request) -> bool:
    # no
    return request.headers.get(PROFILE_ENABLED_HEADER) == '1' or _to_bool(_query_value(request, 'profile'))  # noqa: E501


def _start_request_profiler(request: Request) -> PyInstrumentProfiler | None:
    if not _is_profile_enabled(request):
        return None
    profiler = PyInstrumentProfiler()
    profiler.start()
    return profiler


def _finish_request_profiler(request_profiler: PyInstrumentProfiler | None) -> None:
    if request_profiler is None:
        return
    try:
        request_profiler.stop()
        html_output = request_profiler.output_html()
        LATEST_PROFILE_PATH.write_text(html_output, encoding='utf-8')
    except Exception as e:
        print(f'Failed to persist profile output: {e}')


async def _run_detection(img: bytes, profiled_request: bool) -> DetectionResult:
    # Keep normal traffic off the event loop, but run profiled calls inline so pyinstrument
    # can capture deeper Python call stacks instead of just asyncio.to_thread wait time.
    if profiled_request:
        return detect_spoof_service(img)
    return await asyncio.to_thread(detect_spoof_service, img)


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
    request_profiler = _start_request_profiler(request)
    try:
        print('Processing /detect request...')
        img = get_image(request)
        print('image extracted from request')
        if img is None:
            return Response(
                status_code=HTTPStatus.BAD_REQUEST.value,
                headers=headers,
                description='{"code": "ERR_NO_FILE_UPLOADED"}'
            )

        try:
            print_memory_usage('Starting prediction')
            result = await _run_detection(img, request_profiler is not None)

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
    finally:
        _finish_request_profiler(request_profiler)


@router.post('/detect/verbose')
async def detect_spoof_verbose(request: Request):
    """Endpoint to detect spoofing and return 204 if live, 401 if spoof, and 400 for errors"""

    request_profiler = _start_request_profiler(request)
    try:
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
            result = await _run_detection(img, request_profiler is not None)
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
    finally:
        _finish_request_profiler(request_profiler)
