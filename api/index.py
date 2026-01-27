import os
import threading
import time
from typing import Dict, Iterable, Tuple

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from streamlit.web.bootstrap import run as streamlit_run

_STREAMLIT_PORT = int(os.environ.get("STREAMLIT_SERVER_PORT", "8501"))
_STREAMLIT_STARTED = False
_STREAMLIT_LOCK = threading.Lock()


def _start_streamlit() -> None:
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(_STREAMLIT_PORT))
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
    streamlit_run("app.py", command_line="", args=[], flag_options={})


def _ensure_streamlit_started() -> None:
    global _STREAMLIT_STARTED
    if _STREAMLIT_STARTED:
        return
    with _STREAMLIT_LOCK:
        if _STREAMLIT_STARTED:
            return
        thread = threading.Thread(target=_start_streamlit, daemon=True)
        thread.start()
        for _ in range(50):
            try:
                httpx.get(f"http://127.0.0.1:{_STREAMLIT_PORT}/", timeout=1.0)
                _STREAMLIT_STARTED = True
                return
            except httpx.HTTPError:
                time.sleep(0.1)
        _STREAMLIT_STARTED = True


def _filter_headers(headers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {key: value for key, value in headers if key.lower() not in hop_by_hop}


async def _proxy(request: Request) -> Response:
    _ensure_streamlit_started()
    upstream_url = httpx.URL(
        f"http://127.0.0.1:{_STREAMLIT_PORT}{request.url.path}"
    ).copy_with(query=request.url.query.encode("utf-8"))
    headers = _filter_headers(request.headers.items())
    async with httpx.AsyncClient() as client:
        upstream_response = await client.request(
            request.method,
            upstream_url,
            content=await request.body(),
            headers=headers,
            timeout=None,
        )
    response_headers = _filter_headers(upstream_response.headers.items())
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


app = Starlette(
    routes=[
        Route("/{path:path}", _proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]),
    ]
)
