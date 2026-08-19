import warnings
from io import BytesIO
from pathlib import PurePath

from PIL import Image, ImageOps, UnidentifiedImageError

from app.schedule_ocr.errors import decode_failed, unsupported_image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
SUPPORTED_FORMATS = {
    "PNG": ({"image/png"}, {".png"}),
    "JPEG": ({"image/jpeg"}, {".jpg", ".jpeg"}),
}


def _signature_format(data: bytes) -> str | None:
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    return None


def decode_image(
    data: bytes,
    *,
    content_type: str | None,
    filename: str | None,
    min_width: int,
    min_height: int,
    max_pixels: int,
) -> Image.Image:
    signature_format = _signature_format(data)
    if signature_format is None:
        raise unsupported_image("JPEG 또는 PNG signature가 필요합니다.")
    allowed_content_types, allowed_extensions = SUPPORTED_FORMATS[signature_format]
    if content_type not in allowed_content_types:
        raise unsupported_image("Content-Type과 실제 이미지 signature가 일치하지 않습니다.")
    if not filename or PurePath(filename).suffix.lower() not in allowed_extensions:
        raise unsupported_image("파일 확장자와 실제 이미지 형식이 일치하지 않습니다.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                if probe.format != signature_format:
                    raise unsupported_image("이미지 decoder 형식과 signature가 일치하지 않습니다.")
                width, height = probe.size
                if width < min_width or height < min_height:
                    raise unsupported_image("이미지 해상도가 최소 기준보다 작습니다.")
                if width * height > max_pixels:
                    raise unsupported_image("이미지 pixel 수가 최대 기준을 초과합니다.")
                probe.verify()

            with Image.open(BytesIO(data)) as reopened:
                if reopened.format != signature_format:
                    raise decode_failed()
                reopened.load()
                normalized = ImageOps.exif_transpose(reopened).convert("RGB")
                return normalized.copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise decode_failed() from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
