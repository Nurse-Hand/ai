from io import BytesIO
from pathlib import PurePath
import struct

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


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    offset = 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            return None
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length < 7:
                return None
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            return width, height
        offset += length
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

    dimensions = _png_dimensions(data) if signature_format == "PNG" else _jpeg_dimensions(data)
    if dimensions is None:
        raise decode_failed()
    width, height = dimensions
    if width < min_width or height < min_height:
        raise unsupported_image("이미지 해상도가 최소 기준보다 작습니다.")
    if width * height > max_pixels:
        raise unsupported_image("이미지 pixel 수가 최대 기준을 초과합니다.")

    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format != signature_format or probe.size != dimensions:
                raise unsupported_image("이미지 decoder 형식과 signature가 일치하지 않습니다.")
            probe.verify()

        with Image.open(BytesIO(data)) as reopened:
            if reopened.format != signature_format:
                raise decode_failed()
            reopened.load()
            normalized = ImageOps.exif_transpose(reopened).convert("RGB")
            return normalized.copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise decode_failed() from exc
