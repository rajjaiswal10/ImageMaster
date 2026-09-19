import os
from pathlib import Path

import torch

# ONNX Runtime needs the CUDA DLLs bundled with the CUDA-enabled Torch wheel.
_TORCH_DLL_DIR = None
if os.name == "nt":
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        _TORCH_DLL_DIR = os.add_dll_directory(str(torch_lib))

from PIL import Image, ImageFilter
from rembg import remove, new_session
import onnxruntime as ort

_SESSION = None


def crop_transparent_content(image: Image.Image) -> Image.Image:
    """Trim all fully transparent margins while preserving the visible content."""
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
    elif image.mode == "RGB":
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        image = rgba
    elif image.mode == "L":
        alpha = image
    elif image.mode == "P":
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        image = rgba
    else:
        return image

    bbox = alpha.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def _get_session():
    global _SESSION
    if _SESSION is None:
        # U2NETP is much lighter than the full U2NET and is a good starting point
        # for low-VRAM GPUs. Change to "u2net" for higher quality if your GPU allows.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in providers if provider in available]
        _SESSION = new_session("u2netp", providers=providers or ["CPUExecutionProvider"])
    return _SESSION


def remove_background(src: Path, dst: Path):
    image = Image.open(src).convert("RGBA")
    result = remove(image, session=_get_session(), alpha_matting=True)
    # Keep the model's soft alpha values, while smoothing one-pixel contour
    # steps that can appear in the final PNG edge.
    alpha = result.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.7))
    result.putalpha(alpha)
    result = crop_transparent_content(result)
    result.save(dst, "PNG")
