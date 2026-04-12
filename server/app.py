"""Server entry module expected by OpenEnv validators.

This module exposes:
- `app`  : ASGI app object for uvicorn/gunicorn
- `main` : callable entrypoint function
"""

import os

import uvicorn

from app import app


def main() -> None:
    """Run the API server (validator-compatible entrypoint)."""

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    main()