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


def _decontaminate_edges(crop_bgr, alpha, bg_crop):
    """Remove background color bleed from semi-transparent edge pixels."""
    alpha_f = alpha.astype(np.float32) / 255.0
    edge = (alpha_f > 0.02) & (alpha_f < 0.98)
    if not np.any(edge):
        return crop_bgr

    crop_f = crop_bgr.astype(np.float32)
    denom = np.clip(alpha_f, 0.35, 1.0)[..., None]
    decontaminated = (crop_f - (1 - alpha_f[..., None]) * bg_crop) / denom
    result = np.where(edge[..., None], np.clip(decontaminated, 0, 255), crop_f)
    return result.astype(np.uint8)


def _normalize_object_crop(rgba, target_size):
    h, w = rgba.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas


def extract_all_objects(src: Path, output_dir: Path):
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    bg_plane = _fit_background_plane(image)
    mask = _foreground_mask(image, bg_plane)
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

        # Shrink the mask by a pixel before feathering so background-tinted
        # rim pixels (source of green/color fringing) aren't kept as "inside".
        obj_mask = cv2.erode(obj_mask, np.ones((3, 3), np.uint8), iterations=1)
        if cv2.countNonZero(obj_mask) < min_area:
            continue

        alpha = feather_alpha(obj_mask, feather=4)
        crop_bgr = image[y1:y2, x1:x2]
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
        normalized = _normalize_object_crop(rgba, TARGET_SIZE)
        Image.fromarray(normalized).save(out, "PNG")

        objects.append({
            "id": int(len(objects) + 1),
            "label": "object",
            "confidence": float(0.99),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "url": f"/api/output/{filename}",
        })

    return objects
