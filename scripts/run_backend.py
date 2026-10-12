"""Run the browser backend with a controllable, graceful Uvicorn lifecycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
    app.state.uvicorn_server = server
    server.run()


if __name__ == "__main__":
    main()
