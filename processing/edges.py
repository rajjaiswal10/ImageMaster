import cv2
import numpy as np


def feather_alpha(mask: np.ndarray, feather: int = 4) -> np.ndarray:
    """Convert a hard binary mask into a smooth, anti-aliased alpha channel.

    Builds a signed distance field around the mask boundary so the alpha value
    ramps gradually across `feather` pixels instead of cutting off abruptly.
    """
    binary = (mask > 127).astype(np.uint8) * 255
    if feather <= 0:
        return binary

    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    signed = dist_in - dist_out  # positive inside the object, negative outside

    alpha = np.clip((signed + feather) / (2 * feather) * 255.0, 0, 255)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
    return np.clip(alpha, 0, 255).astype(np.uint8)
