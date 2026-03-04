from __future__ import annotations

import enum


class HTTPStatus(enum.Enum):
    OK = 200
    OK_NO_CONTENT = 204
    CREATED = 201
    PAYMENT_REQUIRED = 402
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    TOO_MANY_REQUESTS = 429
    FORBIDDEN = 403


class SpoofVerboseHTTPStatus(enum.Enum):
    NO_FACE = 405
    MULTIPLE_FACES = 406
    NOT_FRONTAL = 407
    SPOOF_DETECTED = 401
