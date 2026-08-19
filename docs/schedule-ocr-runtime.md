# Schedule OCR runtime verification

## Reproducible inputs

- Platform: `linux/amd64`.
- Base image and digest: `ocr-components.lock`.
- Debian package source: `snapshot.debian.org` at `20260803T000000Z`, with both bookworm
  and bookworm-security and `check-valid-until=no`.
- Pillow is installed separately with `--no-deps --only-binary=:all: --require-hashes`.
- Tesseract and English tessdata package versions are exact apt constraints.
- The image build records the resolved Tesseract, tessdata, libtesseract and Leptonica
  versions and verifies the installed `eng.traineddata` digest.

The snapshot Release URLs and the official Pillow wheel URL returned HTTP 200 on
2026-08-19. The Debian package and extracted model hashes in `ocr-components.lock` were
computed from that fixed snapshot. Docker was not running in the verification environment,
so an actual image build remains mandatory before engine enablement.

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
