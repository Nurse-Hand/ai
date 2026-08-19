import argparse
import json
from pathlib import Path

from main import app

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "openapi" / "internal-ai-v1.json"
INTERNAL_PATHS = {
    "/internal/v1/tasks/extract",
    "/internal/v1/tasks/prioritize",
    "/internal/v1/handoffs/precheck",
    "/internal/v1/handoffs/generate",
    "/internal/v1/audio/analyze",
}


def render_openapi() -> str:
    schema = app.openapi()
    schema["paths"] = {
        path: schema["paths"][path] for path in sorted(INTERNAL_PATHS)
    }
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
