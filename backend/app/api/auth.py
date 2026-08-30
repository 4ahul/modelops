"""API key authentication and per-key rate limiting.

Keys are compared as SHA-256 hashes with a constant-time comparison, and only
hashes are ever configured or stored. A leaked environment dump or a database
backup therefore does not hand over usable credentials.

Rate limiting is a fixed window in Redis, so the limit holds across replicas
rather than per process. When Redis is unavailable the limiter fails **open** and
says so loudly — the alternative is that a monitoring outage becomes a total
outage, which is a worse failure for a service on the request path.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

_RATE_KEY = "modelops:ratelimit:{key_hash}:{window}"


def hash_key(api_key: str) -> str:
    """SHA-256 of an API key, hex-encoded."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_key(api_key: str, valid_hashes: frozenset[str]) -> str | None:
    """Return the matching hash, or ``None``.

    Compared with :func:`hmac.compare_digest` against every candidate. A plain
    ``in`` test on a set is fast but its timing depends on the value, which is
    exactly the signal a key-guessing attack needs.
    """
    candidate = hash_key(api_key)
    matched: str | None = None
    for known in valid_hashes:
        if hmac.compare_digest(candidate, known):
            matched = known
    return matched


class RateLimiter:
    """Fixed-window request counter.

    A fixed window is used rather than a sliding one deliberately: it costs one
    Redis increment per request, and the failure mode — up to 2x the limit across
    a window boundary — is acceptable for protecting a paid API, whereas the
    extra round trips of a sliding window are not.
    """

    def __init__(self, redis: Any | None, *, per_minute: int = 120) -> None:
        self.redis = redis
        self.per_minute = per_minute
        self._local: dict[tuple[str, int], int] = {}
        self.degraded = False

    async def check(self, key_hash: str) -> tuple[bool, int]:
        """Count one request. Returns ``(allowed, remaining)``."""
        if self.per_minute <= 0:
            return True, -1
        window = int(time.time() // 60)

        if self.redis is not None:
            try:
                redis_key = _RATE_KEY.format(key_hash=key_hash, window=window)
                count = int(await self.redis.incr(redis_key))
                if count == 1:
                    # Expiry set only on creation, so a burst cannot keep
                    # pushing the window's end further out.
                    await self.redis.expire(redis_key, 120)
                return count <= self.per_minute, max(0, self.per_minute - count)
            except Exception as exc:
                if not self.degraded:
                    log.warning(
                        "rate_limiter_degraded",
                        error=str(exc),
                        effect="falling back to per-process counters",
                    )
                self.degraded = True

        # Per-process fallback. Weaker with several replicas, but a limit that
        # is only approximately enforced beats refusing all traffic.
        self._local = {k: v for k, v in self._local.items() if k[1] >= window - 1}
        counter = (key_hash, window)
        self._local[counter] = self._local.get(counter, 0) + 1
        count = self._local[counter]
        return count <= self.per_minute, max(0, self.per_minute - count)


class AuthContext:
    """The authenticated caller."""

    def __init__(self, key_hash: str | None, *, anonymous: bool = False) -> None:
        self.key_hash = key_hash
        self.anonymous = anonymous


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_app_settings),
) -> AuthContext:
    """Authenticate a request and apply its rate limit.

    With no keys configured the service runs open and logs a warning on each
    request. That is a legitimate local-development mode and an unacceptable
    production one, which is why
    :meth:`~app.core.config.Settings.validate_for_production` refuses to start a
    production deployment in that state.
    """
    valid = settings.api_key_hash_set
    if not valid:
        log.warning(
            "unauthenticated_request", path=request.url.path, environment=settings.environment
        )
        return AuthContext(None, anonymous=True)

    presented = x_api_key
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            presented = token.strip()

    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_hash = verify_key(presented, valid)
    if key_hash is None:
        # No hint about why: distinguishing "unknown key" from "revoked key"
        # tells an attacker which guesses were close.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        allowed, remaining = await limiter.check(key_hash)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit of {limiter.per_minute} requests/minute exceeded",
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            )
        request.state.rate_limit_remaining = remaining

    return AuthContext(key_hash)


__all__ = ["AuthContext", "RateLimiter", "hash_key", "require_api_key", "verify_key"]
