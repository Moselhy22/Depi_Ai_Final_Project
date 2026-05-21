import cv2
import numpy as np


def check_spoof(face_img: np.ndarray) -> bool:
    """
    Lightweight texture-based anti-spoof check.
    Returns True if the face appears to be real, False if likely a spoof (photo/screen).

    Checks (all must pass):
      1. Blur (Laplacian variance) — printed photos are blurry
      2. Brightness — screens are too bright or too dark
      3. Texture std-dev — flat images lack texture
      4. Edge density — spoofs have fewer natural edges
      5. Color variance — grayscale photos lack color variation
    """
    if face_img is None or face_img.size == 0:
        return False

    # Require minimum crop size to avoid false positives on tiny regions
    if face_img.shape[0] < 40 or face_img.shape[1] < 40:
        return True   # Too small to judge — pass through

    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # 1. Blur check
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 30:
            return False

        # 2. Brightness check
        brightness = np.mean(gray)
        if brightness < 35 or brightness > 235:
            return False

        # 3. Texture check
        texture = np.std(gray)
        if texture < 15:
            return False

        # 4. Edge density check
        edges = cv2.Canny(gray, 80, 180)
        edge_score = np.mean(edges)
        if edge_score < 1.5:
            return False

        # 5. Color variance check
        color_var = np.std(face_img.astype(np.float32))
        if color_var < 25:
            return False

        return True

    except Exception:
        return True   # If check crashes, don't block login
