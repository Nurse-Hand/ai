FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
WORKDIR /app
RUN printf '%s\n' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260803T000000Z/ bookworm main' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20260803T000000Z/ bookworm-security main' \
      > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install --no-install-recommends -y \
      tesseract-ocr=5.3.0-2 \
      tesseract-ocr-eng=1:4.1.0-2 \
      libtesseract5=5.3.0-2 \
      liblept5=1.82.0-3+b3 \
    && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' \
      tesseract-ocr tesseract-ocr-eng libtesseract5 liblept5 > /usr/share/nurse-hand-ocr-components.lock \
    && sha256sum /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
      >> /usr/share/nurse-hand-ocr-components.lock \
    && echo '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2  /usr/share/tesseract-ocr/5/tessdata/eng.traineddata' \
      | sha256sum -c - \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r requirements-ocr.txt \
    && python -c "from PIL import features; assert features.check_feature('libimagequant') is False; assert features.check_feature('raqm') is False" \
    && python -c "from io import BytesIO; from PIL import Image; [Image.open(BytesIO((lambda b: (Image.new('RGB',(32,32),'white').save(b, f), b.getvalue())[1])(BytesIO()))).load() for f in ('PNG','JPEG')]" \
    && python -c "from pathlib import Path; root=Path(__import__('PIL').__file__).parent.parent; names=[p.name.lower() for p in root.rglob('*') if p.is_file()]; assert not any(any(token in name for token in ('imagequant','fribidi','raqm')) for name in names)" \
    && find /usr/local/lib/python3.11/site-packages/PIL -type f -name '*.so' -exec ldd {} \; \
      > /usr/share/pillow-ldd.txt \
    && ! grep -Eiq 'imagequant|fribidi|raqm' /usr/share/pillow-ldd.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
