from pathlib import Path
import hashlib
from importlib.metadata import version
import subprocess
import tarfile
from io import BytesIO

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
    assert "libtesseract5=5.3.0-2" in dockerfile
    assert "liblept5=1.82.0-3+b3" in dockerfile
    assert "--no-deps --only-binary=:all: --require-hashes" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert "rm -f /usr/local/lib/python3.11/site-packages/PIL/_imagingft*.so" in dockerfile
    assert "iq=features.check_feature('libimagequant')" in dockerfile
    assert "assert iq is False" in dockerfile
    assert "rq=features.check_feature('raqm')" in dockerfile
    assert "assert rq is None" in dockerfile
    assert "for f in ('PNG','JPEG')" in dockerfile
    assert "grep -Eiq 'imagequant|fribidi|raqm'" in dockerfile


def test_required_ocr_notices_and_upstream_inventory_are_preserved() -> None:
    required = [
        "THIRD_PARTY_NOTICES.md",
        "ocr-components.lock",
        "licenses/Pillow-12.3.0-LICENSE.txt",
        "licenses/Pillow-12.3.0.cdx.json",
        "licenses/Pillow-12.3.0-auditwheel.cdx.json",
        "licenses/Pillow-12.3.0-wheel-files.txt",
        "licenses/LGPL-2.1-or-later.txt",
        "licenses/Tesseract-5.3.0-LICENSE.txt",
        "licenses/tessdata-4.1.0-LICENSE.txt",
        "licenses/Leptonica-1.82.0-LICENSE.txt",
    ]
    assert all((ROOT / path).is_file() for path in required)
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "LGPL-2.1-or-later" in notices
    assert "GPL-3.0-or-later" in notices
    assert "not a legal" in notices
    assert "not evidence that every optional component is bundled" in notices
    assert "Vendored raqm code" in notices
    assert "removes the `_imagingft`" in notices
    assert "Debian `libfribidi0`" in notices
    assert "pillow_sdist_sha256=3b8182a766685eaa" in (ROOT / "ocr-components.lock").read_text(encoding="utf-8")


def test_leptonica_license_source_and_image_notice_are_hash_locked() -> None:
    license_bytes = (ROOT / "licenses" / "Leptonica-1.82.0-LICENSE.txt").read_bytes()
    digest = hashlib.sha256(license_bytes).hexdigest()
    assert digest == "4d3065116f182e29760af0c901d5dbb2e1e16c42765dfc24e69b26805e2acb1e"

    lock = (ROOT / "ocr-components.lock").read_text(encoding="utf-8")
    assert f"leptonica_preserved_license_sha256={digest}" in lock
    assert "leptonica_upstream_license_download_sha256=87829abb5bbb00b55a107365da89e9a33" in lock
    assert "leptonica_upstream_version=1.82.0" in lock
    assert "leptonica_upstream_source_sha256=40fa9ac1e815b91e0fa73f0737e60c9e" in lock
    assert "liblept5_debian_copyright_sha256=cff4f0cb5db14528a8b84f4a3389012d" in lock

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "cff4f0cb5db14528a8b84f4a3389012d5c6e0f5a75a882509367b2147c05a83e  /usr/share/doc/liblept5/copyright" in dockerfile

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "## Leptonica 1.82.0" in notices
    assert "binary distributions" in notices
    assert "full-image SBOM" in notices


def test_preserved_pillow_license_matches_fixed_linux_wheel_hash() -> None:
    license_bytes = (ROOT / "licenses" / "Pillow-12.3.0-LICENSE.txt").read_bytes()
    digest = hashlib.sha256(license_bytes).hexdigest()
    assert digest == "dda12a98c1979cf3d94df1cff45d27a4cb3f04a60c76f76902ac54cac03ec0ce"
    lock = (ROOT / "ocr-components.lock").read_text(encoding="utf-8")
    assert f"pillow_cp311_manylinux_x86_64_embedded_license_sha256={digest}" in lock
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert f"{digest}  /usr/local/lib/python3.11/site-packages/pillow-12.3.0.dist-info/licenses/LICENSE" in dockerfile


def test_ocr_api_runtime_versions_are_isolated_and_fixed() -> None:
    pins = (ROOT / "requirements-ocr-api.txt").read_text(encoding="utf-8")
    assert "fastapi==0.141.1" in pins
    assert "pydantic==2.13.4" in pins
    assert "starlette==1.6.0" in pins
    assert "python-multipart==0.0.32" in pins
    assert version("fastapi") == "0.141.1"
    assert version("pydantic") == "2.13.4"
    assert version("starlette") == "1.6.0"
    assert version("python-multipart") == "0.0.32"
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements.txt requirements-ocr-api.txt requirements-ocr.txt ./" in dockerfile
    assert "pip install --no-cache-dir -r requirements-ocr-api.txt" in dockerfile


def test_fresh_git_archive_preserves_pillow_license_bytes() -> None:
    tree = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "write-tree"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "archive", tree],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=BytesIO(completed.stdout), mode="r:") as archive:
        archived = archive.extractfile("licenses/Pillow-12.3.0-LICENSE.txt")
        assert archived is not None
        license_bytes = archived.read()
    assert b"\r\n" not in license_bytes
    assert hashlib.sha256(license_bytes).hexdigest() == (
        "dda12a98c1979cf3d94df1cff45d27a4cb3f04a60c76f76902ac54cac03ec0ce"
    )
    attributes = subprocess.run(
        [
            "git", "-c", f"safe.directory={ROOT.as_posix()}", "check-attr", "--cached", "eol", "--",
            "licenses/Pillow-12.3.0-LICENSE.txt", "ocr-components.lock", "Dockerfile",
            "openapi/schedule-ocr.v1.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert all(line.endswith("eol: lf") for line in attributes.splitlines())
