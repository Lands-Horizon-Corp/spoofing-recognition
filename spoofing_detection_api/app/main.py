from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse

from app.api.base_routes import base_route
from app.core import startup
from app.core.config import model_config
from app.core.config import settings
from app.core.middleware import resolve_origin
from app.core.security import limiter
from robyn import Request
from robyn import Response
from robyn import Robyn
from spoofdet.utils.verify_memory import print_memory_usage

print_memory_usage('Starting up the API...')

app = Robyn(__file__, openapi_file_path=settings.OPENAPI_PATH)

origin = settings.CORS_ALLOW_ORIGINS
print(f'Allowed CORS origins: {origin}')

ALLOWED_HEADERS = (
    'Content-Type, Accept, Authorization, Location, '
    'X-Organization-Id, X-User-Agent, X-Device-Type, X-CSRF-Token'
)
ALLOWED_METHODS = 'GET, POST, OPTIONS'
PROFILE_REQUEST_ID_HEADER = 'x-profile-request-id'
PROFILE_ENABLED_HEADER = 'x-profile-enabled'

LATEST_PROFILE_PATH = Path('./profile_latest.html')


# TODO : move to util file
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

    # Fallback path for cases where query_params is missing/None.
    try:
        raw_url = str(getattr(req, 'url', ''))
        parsed_qs = parse_qs(urlparse(raw_url).query)
        values = parsed_qs.get(key)
        return values[0] if values else None
    except Exception:
        return None


@app.startup_handler
async def run_on_startup():
    await startup.download_models()
    print_memory_usage('Startup Complete')
    print('Starting up the API...')
    print_memory_usage('Startup Tasks Completed')
    model_config.load_model_params()


@app.shutdown_handler
async def run_on_shutdown():
    print('Shutting down the API...')


@app.before_request()
def intercept_and_limit(req: Request):
    print(f'Incoming request: {req.method} {req}')

    req_origin = req.headers.get('origin') or ''
    matched_origin = resolve_origin(req_origin)
    print(f"Matched origin: '{matched_origin}'",
          f" for request origin: '{req_origin}'")

    if req.method == 'OPTIONS':
        return Response(
            status_code=204,
            description='',
            headers={
                'Access-Control-Allow-Origin': matched_origin,
                'Access-Control-Allow-Headers': ALLOWED_HEADERS,
                'Access-Control-Allow-Methods': ALLOWED_METHODS,
                'Access-Control-Allow-Credentials': 'true',
                'Access-Control-Max-Age': '86400',
            },
        )

    profile_raw = _query_value(req, 'profile')
    profile_enabled = settings.PROFILING and _to_bool(profile_raw)
    print(f'Profile flag raw={profile_raw} enabled={profile_enabled}')

    if profile_enabled:
        print('Profiling enabled for this request.')
        req.headers.set(PROFILE_ENABLED_HEADER, '1')
        return req
    return limiter.handle_request(app, req)


@app.after_request()
def profile_request(request: Request, response: Response):
    return response

# TODO: separate routes


@app.get('/api/v1/profile/dump')
async def dump_profile(request: Request):
    html_output = None
    if LATEST_PROFILE_PATH.exists():
        try:
            html_output = LATEST_PROFILE_PATH.read_text(encoding='utf-8')
        except Exception:
            html_output = None

    if html_output is None:
        return Response(
            status_code=404,
            headers={'Content-Type': 'application/json'},
            description='{"error": "No request profile has been captured yet. Use ?profile=1 on an endpoint first."}',  # noqa: E501
        )

    with open('./profile.html', 'w', encoding='utf-8') as f:
        f.write(html_output)

    return Response(
        status_code=200,
        headers={'Content-Type': 'text/html'},
        description=html_output,
    )


app.include_router(base_route)

if __name__ == '__main__':
    app.start(host='0.0.0.0', port=8001)
