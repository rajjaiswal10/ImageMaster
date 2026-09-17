from pathlib import Path
from PIL import Image
from rembg import remove, new_session

_SESSION = None

def _get_session():
    global _SESSION
    if _SESSION is None:
        # U2NETP is much lighter than the full U2NET and is a good starting point
        # for low-VRAM GPUs. Change to "u2net" for higher quality if your GPU allows.
        _SESSION = new_session("u2netp")
    return _SESSION

def remove_background(src: Path, dst: Path):
    image = Image.open(src).convert("RGBA")
    result = remove(image, session=_get_session(), alpha_matting=True)
    result.save(dst, "PNG")
