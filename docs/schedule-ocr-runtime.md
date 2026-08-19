# Schedule OCR runtime verification

## Reproducible inputs

- Platform: `linux/amd64`.
- Base image and digest: `ocr-components.lock`.
- Debian package source: `snapshot.debian.org` at `20260803T000000Z`, with both bookworm
  and bookworm-security and `check-valid-until=no`.
- Pillow is installed separately with `--no-deps --only-binary=:all: --require-hashes`.
- The selected wheel archive list and auditwheel SBOM are preserved under `licenses/`.
- The official Pillow 12.3.0 source distribution is
  `https://files.pythonhosted.org/packages/1c/3d/bb7fca845737cf9d7dbde16ed1843984665ff2e0a518f5db43e77ec540b9/pillow-12.3.0.tar.gz`
  with SHA-256 `3b8182a766685eaa002637e28b4ec8d6b18819a0c71f579bf0dbaa5830297cce`.
- Pillow's official build documentation says distributed binaries do not enable GPLv3
  libimagequant. The image build independently asserts `libimagequant=False`, `raqm=False`,
  absence of libimagequant/fribidi/raqm archive names, and absence from `ldd` output.
- The build also decodes generated PNG and JPEG bytes as a core Pillow smoke test.
- Tesseract and English tessdata package versions are exact apt constraints.
- The image build records the resolved Tesseract, tessdata, libtesseract and Leptonica
  versions and verifies the installed `eng.traineddata` digest.

The snapshot Release URLs and the official Pillow wheel URL returned HTTP 200 on
2026-08-19. The Debian package and extracted model hashes in `ocr-components.lock` were
computed from that fixed snapshot. Docker was not running in the verification environment,
so an actual image build remains mandatory before engine enablement.

The package-level Pillow SBOM includes optional source/build capabilities and must not be
read as the selected wheel archive inventory. `Pillow-12.3.0-wheel-files.txt` records the
actual fixed wheel members, and `Pillow-12.3.0-auditwheel.cdx.json` records auditwheel's
binary inventory. The archive contained neither a libimagequant nor a fribidi shared object.
Vendored raqm and the LGPL-2.1-or-later fribidi shim are compiled into `_imagingft`; FriBiDi
itself remains an optional runtime library. Its absence and `raqm=False` are build gates for
this OCR image.

## Required build and inventory gate

```bash
docker buildx build --platform linux/amd64 --sbom=true --provenance=true \
  --tag nurse-hand-ai:schedule-ocr --load .
docker run --rm nurse-hand-ai:schedule-ocr \
  cat /usr/share/nurse-hand-ocr-components.lock
docker sbom --format cyclonedx-json nurse-hand-ai:schedule-ocr \
  > artifacts/nurse-hand-ai-schedule-ocr.cdx.json
syft nurse-hand-ai:schedule-ocr \
  -o spdx-json=artifacts/nurse-hand-ai-schedule-ocr.spdx.json
```

The deployment review must compare the generated lock with `ocr-components.lock`, retain
Debian copyright/common-license paths, retain Pillow's installed `.dist-info/licenses` and
`.dist-info/sboms`, and manually inspect every component whose license or redistribution
obligation is unresolved. Image scanning covers the complete OS/Python/model inventory;
the OCR delta inventory alone is not a complete deployment SBOM.

## Existing API runtime pins

`requirements-ocr-api.txt` fixes the versions of FastAPI, Pydantic, Starlette and
python-multipart exercised by this contract without rewriting the shared `requirements.txt`.
These packages already existed in the AI service dependency graph; this version-fixing delta
does not introduce a new package or model. Their licenses remain part of the required full-image
SBOM review. The separate file only reduces integration conflicts and is not a substitute for a
fully hashed lock of the complete Python environment.
