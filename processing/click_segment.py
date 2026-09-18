import uuid
import numpy as np
import cv2
from PIL import Image
from ultralytics import SAM

from .background import crop_transparent_content
from .edges import feather_alpha

_MODEL = None

# MobileSAM: lightweight promptable segmentation model, auto-downloaded on first use.
MODEL_NAME = "mobile_sam.pt"


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SAM(MODEL_NAME)
    return _MODEL


def _read(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read image")
    return bgr


def mask_to_rgba(image, mask):
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba)


def _predict_mask(path, points):
    """Run MobileSAM with foreground/background point prompts for one object."""
    if not points:
        raise ValueError("No points supplied")

    image = _read(path)
    h, w = image.shape[:2]
    xy = [[int(p["x"]), int(p["y"])] for p in points]
    labels = [int(p.get("label", 1)) for p in points]
    for x, y in xy:
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Click is outside the image")

    model = _model()
    results = model.predict(source=str(path), points=[xy], labels=[labels], verbose=False)
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise ValueError("No object found at that point")

    # MobileSAM can return several candidate masks; the largest is usually the intended object.
    masks = result.masks.data.cpu().numpy()
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best = masks[int(np.argmax(areas))]

    mask = cv2.resize(best.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8) * 255

    # Keep only the component nearest the first (primary) click.
    n, labels_cc, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 2:
        px, py = xy[0]
        label_id = labels_cc[min(py, h - 1), min(px, w - 1)]
        if label_id > 0:
            binary = np.where(labels_cc == label_id, 255, 0).astype(np.uint8)

    return image, feather_alpha(binary, feather=4)


def segment_from_points(path, points):
    """points: list of {x, y, label} for a single object (label 1=foreground, 0=background)."""
    image, alpha = _predict_mask(path, points)
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise ValueError("No object found at that point")

    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    mask = crop_transparent_content(Image.fromarray(alpha))
    return {"bbox": bbox, "mask": mask}


def segment_from_click(path, x, y):
    """Backward-compatible single foreground-point entry point."""
    return segment_from_points(path, [{"x": x, "y": y, "label": 1}])


def export_click_objects(path, objects, output_dir):
    """objects: list of point-lists, one list per selected object."""
    paths = []
    for i, points in enumerate(objects, start=1):
        image, alpha = _predict_mask(path, points)
        rgba = mask_to_rgba(image, alpha)
        rgba = crop_transparent_content(rgba)
        p = output_dir / f"selected_{i:03d}_{uuid.uuid4().hex[:8]}.png"
        rgba.save(p, "PNG")
        paths.append(p)
    return paths
