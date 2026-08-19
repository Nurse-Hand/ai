# OCR third-party notices

This file records technical provenance for the new schedule OCR delta. It is not a legal
opinion and does not assert that the complete deployment image is risk-free. The full image
SBOM and license obligations must be reviewed before production deployment.

## Pillow 12.3.0

- Source: official PyPI CPython 3.11 manylinux x86_64 wheel.
- Project license expression: MIT-CMU.
- Exact wheel digest: see `requirements-ocr.txt` and `ocr-components.lock`.
- Exact embedded LICENSE extracted from the fixed CPython 3.11 manylinux x86_64 wheel:
  `licenses/Pillow-12.3.0-LICENSE.txt`. Its SHA-256 is fixed in `ocr-components.lock` and
  rechecked against the installed wheel during the Docker build.
- Preserved PEP 770 inventory: `licenses/Pillow-12.3.0.cdx.json`.
- Corresponding Pillow 12.3.0 source: official PyPI `pillow-12.3.0.tar.gz`, URL and
  SHA-256 recorded in `ocr-components.lock`.

### Included in the selected wheel

- Pillow/PIL modules under MIT-CMU and the native libraries listed by the preserved
  auditwheel SBOM and wheel file inventory.
- Vendored raqm code under MIT and `fribidi-shim` under LGPL-2.1-or-later are compiled into
  `_imagingft`; the LGPL-2.1-or-later full text is preserved at
  `licenses/LGPL-2.1-or-later.txt`. This notice and corresponding source provenance do not
  by themselves assert that every redistribution obligation has been discharged.
- The deployment image removes the `_imagingft` native extension in the same layer that installs
  the fixed wheel because OCR only requires PNG/JPEG decoding. The original wheel inventory,
  corresponding source and license texts remain preserved for review.

### Optional runtime component

- FriBiDi is loaded by the shim only when a compatible runtime library is available. The
  fixed wheel archive does not contain a `libfribidi` shared library, and the OCR image must
  keep the raqm feature unavailable.
- Debian `libfribidi0` is nevertheless present as a transitive OS package in the Tesseract image;
  removing Pillow `_imagingft` makes Pillow report raqm as unavailable (`None`) while the OS package remains in
  the full-image SBOM and license review scope.

### Absent from the selected wheel

- The fixed wheel archive contains no `libimagequant` shared library and no GPLv3
  (`GPL-3.0-or-later`) libimagequant support. Pillow's official build documentation states that distributed
  binaries do not enable libimagequant.
- The fixed wheel archive contains no `libfribidi` shared library.

The upstream package-level PEP 770 inventory also lists optional build components; it is not evidence that every optional component is bundled in the selected wheel. The wheel archive
list, auditwheel SBOM, build feature checks and final `ldd` output are the controlling image
evidence.

## Tesseract OCR 5.3.0

- Binary package: Debian bookworm `tesseract-ocr=5.3.0-2` from the fixed snapshot.
- Upstream license: Apache-2.0.
- Preserved license: `licenses/Tesseract-5.3.0-LICENSE.txt`.
- Runtime dependencies include exact `libtesseract5=5.3.0-2` and
  Leptonica `liblept5=1.82.0-3+b3` pins. Installed versions are also emitted to
  `/usr/share/nurse-hand-ocr-components.lock` during image build.
- Debian copyright files and `/usr/share/common-licenses` are intentionally not deleted.

## English tessdata 4.1.0

- Data package: Debian bookworm `tesseract-ocr-eng=1:4.1.0-2` from the fixed snapshot.
- Upstream license: Apache-2.0.
- Preserved license: `licenses/tessdata-4.1.0-LICENSE.txt`.
- `eng.traineddata` SHA-256 is fixed and verified during image build.

## Leptonica 1.82.0

- Runtime package: Debian bookworm `liblept5=1.82.0-3+b3` from the fixed snapshot.
- Upstream source: official `DanBloomberg/leptonica` tag `1.82.0`; source archive URL and
  SHA-256 are fixed in `ocr-components.lock`.
- License: Leptonica license, a permissive two-condition redistribution license requiring
  preservation of the copyright notice, conditions and disclaimer in source distributions,
  and reproduction of them in documentation or other materials for binary distributions.
- The upstream license/copyright text is preserved at
  `licenses/Leptonica-1.82.0-LICENSE.txt`. The official raw download digest and the
  repository LF-normalized file digest are separately fixed in `ocr-components.lock`;
  their only byte difference is the raw file's extra terminal blank line.
- The exact Debian package copyright file remains in the image at
  `/usr/share/doc/liblept5/copyright`; its image SHA-256 is fixed and checked during build.
- Leptonica links image format libraries such as libjpeg, libpng, libtiff, libwebp and
  openjpeg. Their actual image packages remain part of the generated full-image SBOM and
  redistribution review; this notice does not clear their separate obligations.

## Deployment gate

Do not enable or deploy the OCR engine when the built image inventory contains
`UNKNOWN`, `NOASSERTION`, custom, noncommercial, or otherwise unresolved redistribution
terms. Known copyleft identifiers require an inclusion/linkage and obligation review; they
are not treated as automatically safe or automatically prohibited. Manual schedule entry
remains the fallback. Existing non-OCR AI dependencies are a separate full-image risk list
and are not reclassified by Issue #38.
