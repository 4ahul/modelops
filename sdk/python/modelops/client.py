"""HTTP client for a ModelOps deployment.

Errors are typed, because the three failure modes need different handling and a
single ``HTTPError`` forces every caller to parse strings:

:class:`NoEligibleModelError`
    The policy excluded every model. Carries the per-model reasons, so the caller
    knows which constraint to relax rather than guessing.

:class:`ProviderFailedError`
    Every candidate was tried and failed. A retry may work; the same request will
    not succeed instantly.

:class:`RateLimitedError`
    Carries ``retry_after``, so a client can back off correctly instead of
    hammering.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_TIMEOUT = 120.0


class ModelOpsError(Exception):
    """Base class for SDK errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(ModelOpsError):
    """The API key was missing, wrong or revoked."""


class RateLimitedError(ModelOpsError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class NoEligibleModelError(ModelOpsError):
    def __init__(self, message: str, excluded: dict[str, str] | None = None) -> None:
        super().__init__(message, status_code=422)
        self.excluded = excluded or {}


class ProviderFailedError(ModelOpsError):
    def __init__(self, message: str, attempts: dict[str, str] | None = None) -> None:
        super().__init__(message, status_code=502)
        self.attempts = attempts or {}


@dataclass
class CompletionResult:
    """A routed completion."""

    content: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    routing_overhead_ms: float
    routing_reason: str
    fallbacks: list[str] = field(default_factory=list)
    task_type: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return self.content

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> CompletionResult:
        return cls(
            content=data["content"],
            model_id=data["model_id"],
            provider=data["provider"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
            routing_overhead_ms=data["routing_overhead_ms"],
            routing_reason=data["routing_reason"],
            fallbacks=list(data.get("fallbacks", [])),
            task_type=data.get("task_type"),
        )


@dataclass
class RouteDecision:
    """What the router would choose, and why."""

    chosen: str
    reason: str
    overhead_ms: float
    candidates: list[dict[str, Any]]
    excluded: dict[str, str]


@dataclass
class EvalReport:
    """Results of an eval run."""

    eval_set: str
    eval_version: int
    examples: int
    duration_s: float
    models: list[dict[str, Any]]
    task_type: str | None = None
    grader: str = "exact_match"
    recommendation: dict[str, Any] | None = None

    def best_by_accuracy(self) -> dict[str, Any] | None:
        return max(self.models, key=lambda m: m["accuracy"], default=None)

    def cheapest_above(self, min_accuracy: float) -> dict[str, Any] | None:
        """The least expensive model clearing an accuracy bar.

        The decision this class exists for: not "which model is best" but "which
        is the cheapest one I can defend switching to".
        """
        eligible = [m for m in self.models if m["accuracy"] >= min_accuracy]
        return min(eligible, key=lambda m: m["cost_per_query"], default=None)

    def table(self) -> str:
        header = f"{'MODEL':<16} {'ACC':>7} {'P95 ms':>9} {'$/QUERY':>10}"
        lines = [header, "-" * len(header)]
        for model in sorted(self.models, key=lambda m: m["cost_per_query"]):
            lines.append(
                f"{model['model_id']:<16} {model['accuracy']:>6.1%} "
                f"{model['p95_latency_ms']:>9.0f} {model['cost_per_query']:>10.6f}"
            )
        return "\n".join(lines)


class ModelOps:
    """Async client.

    Args:
        api_url: Base URL of the deployment.
        api_key: Your API key.
        timeout: Request timeout in seconds. Generous by default because a
            completion behind a fallback chain can legitimately take a while.
        max_retries: Retries on 429 and 5xx, with backoff honouring
            ``Retry-After``.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=self.api_url, timeout=timeout, headers=headers
        )
        self._owns_client = client is None

    async def __aenter__(self) -> ModelOps:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------ transport

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send a request, retrying only what is worth retrying.

        Returns the decoded JSON body. Typed helpers below narrow it â€” see
        :meth:`_request_object` and :meth:`_request_list`, which assert the shape
        at the boundary rather than letting an unexpected body propagate as
        ``Any`` into caller code.
        """
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.RequestError as exc:
                last = ModelOpsError(f"Could not reach {self.api_url}: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                raise last from exc

            if response.status_code < 400:
                return response.json()

            error = _translate(response)
            # Only 429 and 5xx are worth repeating: a 4xx will fail identically
            # however many times it is sent.
            if isinstance(error, RateLimitedError) and attempt < self.max_retries:
                await asyncio.sleep(error.retry_after or 0.5 * 2**attempt)
                continue
            if response.status_code >= 500 and attempt < self.max_retries:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            raise error

        raise last or ModelOpsError("request failed")

    async def _request_object(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """A request whose response must be a JSON object."""
        body = await self._request(method, path, **kwargs)
        if not isinstance(body, dict):
            raise ModelOpsError(
                f"Expected a JSON object from {path}, got {type(body).__name__}. "
                "Check the api_url points at a ModelOps deployment."
            )
        return body

    async def _request_list(self, method: str, path: str, **kwargs: Any) -> list[Any]:
        """A request whose response must be a JSON array."""
        body = await self._request(method, path, **kwargs)
        if not isinstance(body, list):
            raise ModelOpsError(
                f"Expected a JSON array from {path}, got {type(body).__name__}. "
                "Check the api_url points at a ModelOps deployment."
            )
        return body

    # ------------------------------------------------------------- requests

    async def complete(
        self,
        prompt: str,
        task_type: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        model_id: str | None = None,
    ) -> CompletionResult:
        """Route a prompt and return the completion."""
        payload: dict[str, Any] = {"prompt": prompt, "temperature": temperature}
        for key, value in (
            ("task_type", task_type),
            ("max_tokens", max_tokens),
            ("system", system),
            ("model_id", model_id),
        ):
            if value is not None:
                payload[key] = value
        return CompletionResult.from_payload(
            await self._request_object("POST", "/complete", json=payload)
        )

    async def route(
        self, prompt: str, task_type: str | None = None, *, expected_output_tokens: int = 512
    ) -> RouteDecision:
        """Ask which model would run this, without paying for it."""
        data = await self._request_object(
            "POST",
            "/route",
            json={
                "prompt": prompt,
                "task_type": task_type,
                "expected_output_tokens": expected_output_tokens,
            },
        )
        return RouteDecision(
            chosen=data["chosen"],
            reason=data["reason"],
            overhead_ms=data["overhead_ms"],
            candidates=data["candidates"],
            excluded=data["excluded"],
        )

    async def upload_eval_set(
        self,
        name: str,
        examples: list[dict[str, Any]],
        *,
        grader: str = "exact_match",
        task_type: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        return await self._request_object(
            "POST",
            "/evals",
            json={
                "name": name,
                "examples": examples,
                "grader": grader,
                "task_type": task_type,
                "description": description,
            },
        )

    async def run_eval(
        self,
        eval_set: str,
        *,
        model_ids: list[str] | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> EvalReport:
        data = await self._request_object(
            "POST",
            "/evals/run",
            json={
                "eval_set": eval_set,
                "model_ids": model_ids,
                "max_tokens": max_tokens,
                "system": system,
            },
        )
        return EvalReport(
            eval_set=data["eval_set"],
            eval_version=data["eval_version"],
            examples=data["examples"],
            duration_s=data["duration_s"],
            models=data["models"],
            task_type=data.get("task_type"),
            grader=data.get("grader", "exact_match"),
            recommendation=data.get("recommendation"),
        )

    async def eval_history(
        self, *, name: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if name:
            params["name"] = name
        return await self._request_list("GET", "/evals/history", params=params)

    async def metrics(self, *, hours: int = 24, task_type: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"hours": hours}
        if task_type:
            params["task_type"] = task_type
        return await self._request_object("GET", "/metrics", params=params)

    async def models(self) -> dict[str, Any]:
        return await self._request_object("GET", "/models")

    async def alerts(self, *, limit: int = 50, unacknowledged_only: bool = False) -> list[Any]:
        return await self._request_list(
            "GET",
            "/alerts",
            params={"limit": limit, "unacknowledged_only": unacknowledged_only},
        )

    async def health(self) -> dict[str, Any]:
        return await self._request_object("GET", "/health")


class ModelOpsSync:
    """Synchronous client, for scripts and notebooks.

    Each call runs the async client in its own event loop. Fine for scripts;
    inside an existing loop, use :class:`ModelOps` directly rather than nesting
    loops.
    """

    def __init__(self, api_url: str, api_key: str | None = None, **kwargs: Any) -> None:
        self._factory = lambda: ModelOps(api_url, api_key, **kwargs)

    def _run(self, method: str, *args: Any, **kwargs: Any) -> Any:
        async def call() -> Any:
            async with self._factory() as client:
                return await getattr(client, method)(*args, **kwargs)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(call())
        raise RuntimeError(
            "ModelOpsSync cannot be used inside a running event loop. Use ModelOps instead."
        )

    def complete(
        self, prompt: str, task_type: str | None = None, **kwargs: Any
    ) -> CompletionResult:
        return self._run("complete", prompt, task_type, **kwargs)  # type: ignore[no-any-return]

    def route(self, prompt: str, task_type: str | None = None, **kwargs: Any) -> RouteDecision:
        return self._run("route", prompt, task_type, **kwargs)  # type: ignore[no-any-return]

    def run_eval(self, eval_set: str, **kwargs: Any) -> EvalReport:
        return self._run("run_eval", eval_set, **kwargs)  # type: ignore[no-any-return]

    def upload_eval_set(self, name: str, examples: list[dict[str, Any]], **kwargs: Any) -> Any:
        return self._run("upload_eval_set", name, examples, **kwargs)

    def metrics(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("metrics", **kwargs)  # type: ignore[no-any-return]

    def models(self) -> dict[str, Any]:
        return self._run("models")  # type: ignore[no-any-return]

    def health(self) -> dict[str, Any]:
        return self._run("health")  # type: ignore[no-any-return]


def _translate(response: httpx.Response) -> ModelOpsError:
    """Turn an error response into the right exception type."""
    try:
        body = response.json()
    except Exception:
        body = {}

    detail = body.get("detail", response.text[:200]) if isinstance(body, dict) else body
    kind: str | None = None
    context: dict[str, Any] = {}
    message: Any = detail
    # FastAPI nests a dict detail, which is where the router puts its reasons.
    if isinstance(detail, dict):
        kind = detail.get("kind")
        context = detail.get("context") or {}
        message = detail.get("detail", str(detail))

    if response.status_code in (401, 403):
        return AuthenticationError(str(message), status_code=response.status_code)
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        return RateLimitedError(str(message), float(retry_after) if retry_after else None)
    if kind == "no_eligible_model":
        return NoEligibleModelError(str(message), context.get("excluded"))
    if kind in ("all_providers_failed", "provider_failed"):
        return ProviderFailedError(str(message), context.get("attempts"))
    return ModelOpsError(str(message), status_code=response.status_code)


__all__ = [
    "AuthenticationError",
    "CompletionResult",
    "EvalReport",
    "ModelOps",
    "ModelOpsError",
    "ModelOpsSync",
    "NoEligibleModelError",
    "ProviderFailedError",
    "RateLimitedError",
    "RouteDecision",
]
