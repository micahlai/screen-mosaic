import numpy as np
import cv2

def compute_homography(corners_uv, screen_w, screen_h):
    """
    corners_uv: list of 4 [u, v] points (values 0.0-1.0) in master image space
                order: top-left, top-right, bottom-right, bottom-left
    Returns a 3x3 matrix that maps master UV coords -> slave pixel coords
    """
    src = np.array(corners_uv, dtype=np.float32)

    dst = np.array([
        [0,        0],
        [screen_w, 0],
        [screen_w, screen_h],
        [0,        screen_h],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    return H

def warp_master_to_slave(master_image, corners_uv, screen_w, screen_h):
    """
    Crops and warps the master image so this slave's portion fills its screen.
    """
    mh, mw = master_image.shape[:2]

    # Convert UV (0-1) corners to actual pixel coords in the master image
    src_pixels = np.array([[u * mw, v * mh] for u, v in corners_uv], dtype=np.float32)

    dst_pixels = np.array([
        [0,        0],
        [screen_w, 0],
        [screen_w, screen_h],
        [0,        screen_h],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pixels, dst_pixels)
    result = cv2.warpPerspective(master_image, H, (screen_w, screen_h))
    return result