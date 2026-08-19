# OCR third-party notices

This file records technical provenance for the new schedule OCR delta. It is not a legal
opinion and does not assert that the complete deployment image is risk-free. The full image
SBOM and license obligations must be reviewed before production deployment.

## Pillow 12.3.0

- Source: official PyPI CPython 3.11 manylinux x86_64 wheel.
- Project license expression: MIT-CMU.
- Exact wheel digest: see `requirements-ocr.txt` and `ocr-components.lock`.
- Preserved notices: `licenses/Pillow-12.3.0-LICENSE.txt`.
- Preserved PEP 770 inventory: `licenses/Pillow-12.3.0.cdx.json`.

The official wheel inventory includes bundled components under MIT-CMU, LGPL-2.1-or-later,
GPL-3.0-or-later, FTL, MIT, BSD, IJG, libtiff, X11 and Zlib identifiers. In particular,
`fribidi-shim`, `FriBiDi`, `libimagequant`, FreeType, libjpeg-turbo, libpng and zlib-related
notices are retained in the upstream LICENSE and SBOM. Their actual inclusion/linkage and
redistribution obligations must be evaluated against the built linux/amd64 image.

## Tesseract OCR 5.3.0

- Binary package: Debian bookworm `tesseract-ocr=5.3.0-2` from the fixed snapshot.
- Upstream license: Apache-2.0.
- Preserved license: `licenses/Tesseract-5.3.0-LICENSE.txt`.
- Runtime dependencies include `libtesseract5` and Leptonica (`liblept5`). Exact installed
  versions are emitted to `/usr/share/nurse-hand-ocr-components.lock` during image build.
- Debian copyright files and `/usr/share/common-licenses` are intentionally not deleted.

## English tessdata 4.1.0

- Data package: Debian bookworm `tesseract-ocr-eng=1:4.1.0-2` from the fixed snapshot.
- Upstream license: Apache-2.0.
- Preserved license: `licenses/tessdata-4.1.0-LICENSE.txt`.
- `eng.traineddata` SHA-256 is fixed and verified during image build.

## Deployment gate

Do not enable or deploy the OCR engine when the built image inventory contains
`UNKNOWN`, `NOASSERTION`, custom, noncommercial, or otherwise unresolved redistribution
terms. Known copyleft identifiers require an inclusion/linkage and obligation review; they
are not treated as automatically safe or automatically prohibited. Manual schedule entry
remains the fallback. Existing non-OCR AI dependencies are a separate full-image risk list
and are not reclassified by Issue #38.
