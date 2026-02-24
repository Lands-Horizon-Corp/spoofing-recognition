from __future__ import annotations

from app.api.base_routes import base_route
from app.core import startup
from app.core.config import model_config
from app.core.config import settings
from app.core.security import limiter
from robyn import ALLOW_CORS
from robyn import Request
from robyn import Robyn
from spoofdet.verify_memory import print_memory_usage

print_memory_usage('Starting up the API...')


app = Robyn(__file__, openapi_file_path=settings.OPENAPI_PATH)


origin = settings.CORS_ALLOW_ORIGINS
ALLOW_CORS(app, origins=settings.CORS_ALLOW_ORIGINS)


@app.startup_handler
async def run_on_startup():
    await startup.download_model()
    print_memory_usage('Startup Complete')
    print('Starting up the API...')
    print_memory_usage('Startup Tasks Completed')
    model_config.load_model_params()


@app.before_request()
def intercept_and_limit(req: Request):
    return limiter.handle_request(app, req)


app.include_router(base_route)


if __name__ == '__main__':
    app.start(host='0.0.0.0', port=8001)
