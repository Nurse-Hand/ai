from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pillow_runtime_is_hash_locked_for_deployment_wheel() -> None:
    lock = (ROOT / "requirements-ocr.txt").read_text(encoding="utf-8")
    assert "Pillow==12.3.0" in lock
    assert "23d27a3e0307ec2244cc51e7287b919aa68d097504ebe19df4e76a98a3eea5bd" in lock


def test_docker_runtime_uses_digest_snapshot_and_exact_packages() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408e" in dockerfile
    assert "archive/debian/20260803T000000Z/ bookworm main" in dockerfile
    assert "archive/debian-security/20260803T000000Z/ bookworm-security main" in dockerfile
    assert "tesseract-ocr=5.3.0-2" in dockerfile
    assert "tesseract-ocr-eng=1:4.1.0-2" in dockerfile
    assert "--no-deps --only-binary=:all: --require-hashes" in dockerfile
    assert "sha256sum -c -" in dockerfile


def test_required_ocr_notices_and_upstream_inventory_are_preserved() -> None:
    required = [
        "THIRD_PARTY_NOTICES.md",
        "ocr-components.lock",
        "licenses/Pillow-12.3.0-LICENSE.txt",
        "licenses/Pillow-12.3.0.cdx.json",
        "licenses/Tesseract-5.3.0-LICENSE.txt",
        "licenses/tessdata-4.1.0-LICENSE.txt",
    ]
    assert all((ROOT / path).is_file() for path in required)
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "LGPL-2.1-or-later" in notices
    assert "GPL-3.0-or-later" in notices
    assert "not a legal" in notices
