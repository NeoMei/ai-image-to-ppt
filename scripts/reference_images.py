"""Validate bounded local reference images before building provider requests."""
import base64
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image

from image_output import ImageOutputError

MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_REFERENCE_TOTAL_BYTES = 30 * 1024 * 1024
MAX_REFERENCE_IMAGES = 10


@dataclass(frozen=True)
class ReferenceImage:
    mime_type: str
    data: bytes

    def base64_data(self) -> str:
        return base64.b64encode(self.data).decode('ascii')

    def data_url(self) -> str:
        return f'data:{self.mime_type};base64,{self.base64_data()}'


def read_reference_images(paths: Optional[Sequence[str]]) -> List[ReferenceImage]:
    if paths is None:
        return []
    if not isinstance(paths, (list, tuple)) or len(paths) > MAX_REFERENCE_IMAGES:
        raise ImageOutputError('invalid reference image list')
    result, total = [], 0
    for path in paths:
        if not isinstance(path, str) or not path:
            raise ImageOutputError('reference image must be a local file')
        source = Path(path)
        if not source.is_file():
            raise ImageOutputError('reference image must be a local file')
        with source.open('rb') as stream:
            data = stream.read(MAX_REFERENCE_BYTES + 1)
        total += len(data)
        if len(data) > MAX_REFERENCE_BYTES or total > MAX_REFERENCE_TOTAL_BYTES:
            raise ImageOutputError('reference image budget exceeded')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    mime = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}.get(image.format)
                    width, height = image.size
                    if not mime or min(width, height) < 15 or width * height > 36_000_000 or not 1/16 <= width/height <= 16 or getattr(image, 'n_frames', 1) != 1:
                        raise ImageOutputError('unsupported reference image')
                    image.load()
        except Exception as error:
            raise ImageOutputError('invalid reference image') from error
        result.append(ReferenceImage(mime, data))
    return result
