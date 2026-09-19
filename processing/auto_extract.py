
from pathlib import Path
import uuid

from PIL import Image
import cv2
import numpy as np

from .edges import feather_alpha

MIN_AREA_RATIO = 0.0015
BORDER_MARGIN_RATIO = 0.02
PADDING_RATIO = 0.04
TARGET_SIZE = (512, 512)
MIN_DISTANCE_FLOOR = 12  # keep Otsu from collapsing near-zero on very flat/plain backgrounds
CHROMA_TOLERANCE = 85


def _fit_background_plane(image_bgr):
    """Model the background as a linear color gradient fit from the border pixels.

    A single sampled color breaks down on gradient/vignette backgrounds; fitting
    a plane per channel lets the "expected background" vary smoothly across the
    image so gradients aren't mistaken for foreground.
    """
    h, w = image_bgr.shape[:2]
    margin = max(2, int(min(h, w) * BORDER_MARGIN_RATIO))

    yy, xx = np.mgrid[0:h, 0:w]
    border = np.zeros((h, w), dtype=bool)
    border[0:margin, :] = True
    border[h - margin:h, :] = True
    border[:, 0:margin] = True
    border[:, w - margin:w] = True

    ys = yy[border].astype(np.float32)
    xs = xx[border].astype(np.float32)
    colors = image_bgr[border].astype(np.float32)

    design = np.stack([ys, xs, np.ones_like(ys)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, colors, rcond=None)  # rows=[y,x,1], cols=B,G,R

    plane = coeffs[0] * yy[..., None] + coeffs[1] * xx[..., None] + coeffs[2]
    return np.clip(plane, 0, 255).astype(np.float32)


def _foreground_mask(image_bgr, bg_plane):
    h, w = image_bgr.shape[:2]
    border = np.zeros((h, w), dtype=bool)
    margin = max(2, int(min(h, w) * BORDER_MARGIN_RATIO))
    border[0:margin, :] = True
    border[h - margin:h, :] = True
    border[:, 0:margin] = True
    border[:, w - margin:w] = True

    bright_pixels = image_bgr[border]
    border_mean = bright_pixels.mean(axis=0)
    border_std = bright_pixels.std(axis=0)
    border_luma = 0.299 * border_mean[2] + 0.587 * border_mean[1] + 0.114 * border_mean[0]
    border_noise = np.mean(border_std)

    if border_luma > 220 and border_noise < 18:
        bg_color = np.clip(border_mean.astype(np.float32), 0, 255)
        diff = np.abs(image_bgr.astype(np.float32) - bg_color.reshape(1, 1, 3)).max(axis=2)
        mask = (diff > 12).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.clip(bg_plane, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    diff = lab - bg_lab
    # De-emphasize lightness: white/near-white backgrounds vary mostly from
    # lighting/shadow, so weighting L as heavily as color causes false positives.
    dist = np.sqrt((diff[..., 0] * 0.6) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)

    dist_blur = cv2.GaussianBlur(dist, (5, 5), 0)
    dist_u8 = cv2.normalize(dist_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_thresh, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_thresh < MIN_DISTANCE_FLOOR:
        # Near-uniform image; Otsu latched onto noise. Fall back to a small fixed
        # cut so we don't mask out everything (or nothing).
        _, mask = cv2.threshold(dist_u8, MIN_DISTANCE_FLOOR, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _chroma_distance(image_bgr, key_color):
    key_patch = np.uint8([[key_color]])
    key_lab = cv2.cvtColor(key_patch, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(image_lab - key_lab, axis=2)


def _chroma_key_mask(image_bgr, key_color, tolerance=CHROMA_TOLERANCE):
    """Build a foreground mask by removing pixels close to the picked key color."""
    distance = _chroma_distance(image_bgr, key_color)
    mask = (distance > tolerance).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _chroma_key_alpha(image_bgr, key_color, tolerance=CHROMA_TOLERANCE):
    """Return per-pixel transparency so keyed colors disappear inside objects too."""
    distance = _chroma_distance(image_bgr, key_color)
    soft_start = max(0, tolerance - 12)
    alpha = np.clip((distance - soft_start) / max(1, tolerance - soft_start) * 255, 0, 255)
    return alpha.astype(np.uint8)


def _decontaminate_edges(crop_bgr, alpha, bg_crop):
    """Remove background color bleed from semi-transparent edge pixels."""
    alpha_f = alpha.astype(np.float32) / 255.0
    edge = (alpha_f > 0.02) & (alpha_f < 0.98)
    if not np.any(edge):
        return crop_bgr

    crop_f = crop_bgr.astype(np.float32)
    # Full color unmixing is unstable on nearly transparent pixels: a small
    # error in the estimated background gets amplified and can turn a green
    # screen edge magenta. Apply it only toward the opaque side of the rim and
    # blend the correction back into the source at the outer edge.
    correction_strength = np.clip((alpha_f - 0.18) / 0.62, 0.0, 1.0)[..., None]
    denom = np.clip(alpha_f, 0.5, 1.0)[..., None]
    decontaminated = (crop_f - (1 - alpha_f[..., None]) * bg_crop) / denom
    decontaminated = np.clip(decontaminated, 0, 255)
    corrected = crop_f + (decontaminated - crop_f) * correction_strength
    result = np.where(edge[..., None], np.clip(corrected, 0, 255), crop_f)
    return result.astype(np.uint8)


def _refine_object_mask(crop_bgr, object_mask):
    """Refine a color-derived contour before feathering its alpha edge."""
    if cv2.countNonZero(object_mask) == 0:
        return object_mask

    sure_foreground = cv2.erode(object_mask, np.ones((7, 7), np.uint8), iterations=1)
    if cv2.countNonZero(sure_foreground) == 0:
        return object_mask

    grabcut_mask = np.full(object_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[object_mask > 0] = cv2.GC_PR_FGD
    grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            crop_bgr,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return object_mask

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return refined


def _normalize_object_crop(rgba, target_size, border_thickness=0.0):
    h, w = rgba.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Add a one-pixel antialiased outline around the visible object silhouette.
    alpha = resized[:, :, 3]
    if border_thickness > 0:
        binary = (alpha > 127).astype(np.uint8)
        outside_distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        ring = np.clip(
            (border_thickness - outside_distance)
            / max(border_thickness, 0.01)
            * 255,
            0,
            255,
        ).astype(np.uint8)
        ring = cv2.GaussianBlur(ring, (0, 0), 0.55)
        outline = (alpha < 240) & (ring > 0)
        resized[outline, :3] = 0
        resized[outline, 3] = np.maximum(alpha[outline], ring[outline])

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas


def extract_all_objects(
    src: Path,
    output_dir: Path,
    chroma_key=False,
    chroma_color=(0, 255, 0),
    chroma_tolerance=CHROMA_TOLERANCE,
    border_thickness=0.0,
):
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    bg_plane = _fit_background_plane(image)
    mask = _chroma_key_mask(image, chroma_color, chroma_tolerance) if chroma_key else _foreground_mask(image, bg_plane)
    chroma_alpha = _chroma_key_alpha(image, chroma_color, chroma_tolerance) if chroma_key else None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_AREA_RATIO * image.shape[0] * image.shape[1]
    padding = max(1, int(min(image.shape[:2]) * PADDING_RATIO))

    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        obj_mask = contour_mask[y1:y2, x1:x2]
        if obj_mask.size == 0:
            continue

        crop_bgr = image[y1:y2, x1:x2]
        # Refine the contour while the source color is still available. This
        # removes probable background pixels inside a hard color-derived rim.
        obj_mask = _refine_object_mask(crop_bgr, obj_mask)
        obj_mask = cv2.erode(obj_mask, np.ones((3, 3), np.uint8), iterations=1)
        if cv2.countNonZero(obj_mask) < min_area:
            continue

        alpha = feather_alpha(obj_mask, feather=4)
        if chroma_alpha is not None:
            alpha = np.minimum(alpha, chroma_alpha[y1:y2, x1:x2])
        crop_bgr = _decontaminate_edges(crop_bgr, alpha, bg_plane[y1:y2, x1:x2])
        rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha

        # Require a solid opaque core, not just a few stray feathered pixels,
        # otherwise the saved PNG looks like an empty/blank thumbnail.
        opaque_area = int(np.count_nonzero(alpha > 128))
        if opaque_area < min_area:
            continue

        filename = f"{uuid.uuid4().hex}_object.png"
        out = output_dir / filename
        normalized = _normalize_object_crop(rgba, TARGET_SIZE, border_thickness)
        Image.fromarray(normalized).save(out, "PNG")

        objects.append({
            "id": int(len(objects) + 1),
            "label": "object",
            "confidence": float(0.99),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "url": f"/api/output/{filename}",
        })

    return objects
