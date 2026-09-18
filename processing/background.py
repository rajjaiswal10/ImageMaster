from pathlib import Path
from PIL import Image
from rembg import remove, new_session

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
        _SESSION = new_session("u2netp")
    return _SESSION


def remove_background(src: Path, dst: Path):
    image = Image.open(src).convert("RGBA")
    result = remove(image, session=_get_session(), alpha_matting=True)
    result = crop_transparent_content(result)
    result.save(dst, "PNG")
