import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI

from app.routers import audio, handoffs, tasks

ARTIFACT = ROOT / "openapi" / "internal-ai-v1.json"
INTERNAL_PATHS = {
    "/internal/v1/tasks/extract",
    "/internal/v1/tasks/prioritize",
    "/internal/v1/handoffs/precheck",
    "/internal/v1/handoffs/generate",
    "/internal/v1/audio/analyze",
}


def build_internal_app() -> FastAPI:
    internal_app = FastAPI(title="Nurse Hand Internal AI API", version="1.0.0")
    internal_app.include_router(tasks.router)
    internal_app.include_router(handoffs.router)
    internal_app.include_router(audio.router)
    return internal_app


def render_openapi() -> str:
    schema = build_internal_app().openapi()
    schema["paths"] = {path: schema["paths"][path] for path in sorted(INTERNAL_PATHS)}
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_openapi()
    if args.check:
        if not ARTIFACT.exists() or ARTIFACT.read_text(encoding="utf-8") != rendered:
            print(f"OpenAPI drift detected: {ARTIFACT}")
            return 1
        print(f"OpenAPI is up to date: {ARTIFACT}")
        return 0
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(rendered, encoding="utf-8")
    print(f"Generated {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
