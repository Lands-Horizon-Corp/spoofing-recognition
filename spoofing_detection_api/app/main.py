from __future__ import annotations

from app.api.base_routes import base_route
from app.core import startup
from app.core.config import model_config
from app.core.config import settings
from app.core.middleware import resolve_origin
from app.core.security import limiter
from robyn import Request
from robyn import Response
from robyn import Robyn
from spoofdet.verify_memory import print_memory_usage

print_memory_usage('Starting up the API...')


app = Robyn(__file__, openapi_file_path=settings.OPENAPI_PATH)

origin = settings.CORS_ALLOW_ORIGINS
print(f'Allowed CORS origins: {origin}')

# We handle CORS fully in before_request and after_request

ALLOWED_HEADERS = (
    'Content-Type, Accept, Authorization, Location, '
    'X-Organization-Id, X-User-Agent, X-Device-Type, X-CSRF-Token'
)

ALLOWED_METHODS = 'GET, POST, OPTIONS'

# ContextVar to pass origin from before_request → after_request


@app.startup_handler
async def run_on_startup():
    await startup.download_model()
    print_memory_usage('Startup Complete')
    print('Starting up the API...')
    print_memory_usage('Startup Tasks Completed')
    model_config.load_model_params()


@app.before_request()
def intercept_and_limit(req: Request):
    print(f'Incoming request: {req.method} {req}')

    # Stash the matched origin for after_request to use
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
    return limiter.handle_request(app, req)


app.include_router(base_route)

if __name__ == '__main__':
    app.start(host='0.0.0.0', port=8001)
