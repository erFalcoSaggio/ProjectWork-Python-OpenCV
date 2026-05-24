import cv2
import numpy as np

def normale(frame):
    return frame

def grigio(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def negativo(frame):
    return cv2.bitwise_not(frame)

_SEPIA_KERNEL = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])

def sepia(frame):
    result = cv2.transform(frame.astype(np.float64), _SEPIA_KERNEL)
    return np.clip(result, 0, 255).astype(np.uint8)

def heatmap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

def cartoon(frame):
    color = frame
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_3ch)

def pixelate(frame):
    h, w = frame.shape[:2]
    piccolo = cv2.resize(frame, (w // 10, h // 10), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(piccolo, (w, h), interpolation=cv2.INTER_NEAREST)

_vignetta_cache = {"size": None, "mask": None}

def vignetta(frame):
    h, w = frame.shape[:2]
    if _vignetta_cache["size"] != (w, h):
        kx = cv2.getGaussianKernel(w, 0.5 * w)
        ky = cv2.getGaussianKernel(h, 0.5 * h)
        mask = ky * kx.T
        mask = mask / mask.max()
        _vignetta_cache["mask"] = mask[:, :, np.newaxis]
        _vignetta_cache["size"] = (w, h)
    result = frame.astype(np.float64) * _vignetta_cache["mask"]
    return result.astype(np.uint8)

def flip_specchio(frame):
    return cv2.flip(frame, 1)

_MOTION_KERNEL = np.zeros((15, 15))
_MOTION_KERNEL[7, :] = 1.0 / 15

def motion_blur(frame):
    return cv2.filter2D(frame, -1, _MOTION_KERNEL)

def glitch(frame):
    result = frame.copy()
    h = result.shape[0]
    for _ in range(12):
        y = np.random.randint(0, h)
        altezza = np.random.randint(2, 20)
        shift = np.random.randint(-60, 60)
        result[y:y+altezza, :] = np.roll(result[y:y+altezza, :], shift, axis=1)
    b, g, r = cv2.split(result)
    s = np.random.randint(2, 8)
    r = np.roll(r, s, axis=1)
    b = np.roll(b, -s, axis=1)
    return cv2.merge([b, g, r])

def esplosione(frame):
    result = frame.copy()
    h, w = result.shape[:2]
    for _ in range(18):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        r = np.random.randint(10, 80)
        colore = (int(np.random.randint(0, 255)),
                  int(np.random.randint(0, 255)),
                  int(np.random.randint(0, 255)))
        cv2.circle(result, (cx, cy), r, colore, -1)
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), r + 20, colore, 3)
        cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)
    return result

def disco(frame):
    result = frame.copy()
    h, w = result.shape[:2]
    righe, colonne = 6, 8
    cell_h, cell_w = h // righe, w // colonne
    t = (cv2.getTickCount() / cv2.getTickFrequency() * 60) % 180
    for r in range(righe):
        for c in range(colonne):
            hue = int((t + r * 20 + c * 15) % 180)
            colore = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]),
                                  cv2.COLOR_HSV2BGR)[0][0]
            overlay = result.copy()
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colore.tolist(), -1)
            cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)
    return result

def shake(frame):
    h, w = frame.shape[:2]
    dx = np.random.randint(-15, 15)
    dy = np.random.randint(-15, 15)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h))

def statica_tv(frame):
    rumore = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
    return cv2.addWeighted(frame, 0.6, rumore, 0.4, 0)

def zoom_pulse(frame):
    h, w = frame.shape[:2]
    t = cv2.getTickCount() / cv2.getTickFrequency()
    scala = 1.0 + 0.015 * (np.sin(t * 2.0) * 0.5 + 0.5)
    new_w, new_h = int(w * scala), int(h * scala)
    ingrandito = cv2.resize(frame, (new_w, new_h))
    x1 = (new_w - w) // 2
    y1 = (new_h - h) // 2
    return ingrandito[y1:y1+h, x1:x1+w]
