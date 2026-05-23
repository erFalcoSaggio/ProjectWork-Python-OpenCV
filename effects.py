# Falcari - effetti avanzati: face detection, overlay PNG, movimento
import cv2
import numpy as np
import os
import time

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# Testo mostrato dal filtro "Etichetta" sopra ogni faccia rilevata.
ETICHETTA = "Falco"

_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Cache dei PNG già letti dal disco: la chiave è il path, il valore è
# l'immagine BGRA o None se il file non esiste / non è valido.
_png_cache = {}


def _carica_png(path):
    # legge il PNG con alpha solo la prima volta, poi lo prende dalla cache
    if path in _png_cache:
        return _png_cache[path]
    img = None
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3 or img.shape[2] < 4:
            img = None
    _png_cache[path] = img
    return img


def _sovrapponi_png(frame, path, x, y, w, h):
    # disegna un PNG con trasparenza; ritorna False se il PNG non esiste
    if w <= 0 or h <= 0:
        return False
    img = _carica_png(path)
    if img is None:
        return False

    img = cv2.resize(img, (w, h))
    bgr = img[:, :, :3]
    alpha = img[:, :, 3:] / 255.0

    y1, y2 = max(y, 0), min(y + h, frame.shape[0])
    x1, x2 = max(x, 0), min(x + w, frame.shape[1])
    if y1 >= y2 or x1 >= x2:
        return True
    ay1, ay2 = y1 - y, y2 - y
    ax1, ax2 = x1 - x, x2 - x

    roi = frame[y1:y2, x1:x2]
    a = alpha[ay1:ay2, ax1:ax2]
    frame[y1:y2, x1:x2] = (bgr[ay1:ay2, ax1:ax2] * a + roi * (1 - a)).astype(np.uint8)
    return True


# ---------------------------------------------------------------------------
# Effetti con face detection (firma: frame, facce)
# ---------------------------------------------------------------------------

def sfondo_sfocato(frame, facce):
    # sfoca tutto il frame e tiene nitida solo l'area ellittica delle facce
    sfocato = cv2.GaussianBlur(frame, (51, 51), 0)
    if len(facce) == 0:
        return sfocato
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in facce:
        cv2.ellipse(mask, (x + w // 2, y + h // 2),
                    (int(w * 0.6), int(h * 0.6)), 0, 0, 360, 255, -1)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3ch == 255, frame, sfocato)


# --- Cappello (cilindro classico) -----------------------------------------

def _disegna_cilindro(frame, fx, fy, fw, fh):
    # disegna un cappello a cilindro centrato sopra la faccia
    nero = (15, 15, 15)
    rosso_band = (40, 40, 180)

    # tesa: ellisse appiattita poggiata sulla testa
    brim_cx = fx + fw // 2
    brim_cy = fy + int(fh * 0.05)        # poco sopra la fronte
    brim_rx = int(fw * 0.75)
    brim_ry = max(6, int(fh * 0.07))
    cv2.ellipse(frame, (brim_cx, brim_cy), (brim_rx, brim_ry),
                0, 0, 360, nero, -1)

    # corpo del cilindro
    crown_w = int(fw * 0.65)
    crown_h = int(fh * 0.75)
    crown_x = brim_cx - crown_w // 2
    crown_top = brim_cy - crown_h
    cv2.rectangle(frame, (crown_x, crown_top),
                  (crown_x + crown_w, brim_cy), nero, -1)

    # tappo superiore (piccola ellisse) → effetto 3D
    cv2.ellipse(frame, (brim_cx, crown_top),
                (crown_w // 2, max(4, crown_h // 14)),
                0, 0, 360, (35, 35, 35), -1)

    # fascia colorata appena sopra la tesa
    band_h = max(4, int(fh * 0.06))
    cv2.rectangle(frame, (crown_x, brim_cy - band_h),
                  (crown_x + crown_w, brim_cy), rosso_band, -1)


def cappello(frame, facce):
    # cappello PNG sopra la faccia; se manca disegna un cilindro classico
    path = os.path.join(ASSETS, "cappello.png")
    for (fx, fy, fw, fh) in facce:
        hat_w = int(fw * 1.3)
        hat_h = int(fh * 0.7)
        hat_x = fx - (hat_w - fw) // 2
        hat_y = fy - hat_h
        if not _sovrapponi_png(frame, path, hat_x, hat_y, hat_w, hat_h):
            _disegna_cilindro(frame, fx, fy, fw, fh)
    return frame


# --- Occhiali (da sole rotondi) -------------------------------------------

def _disegna_occhiali(frame, cx_l, cx_r, cy, raggio):
    # disegna occhiali rotondi da sole con bridge e aste
    lente = (15, 15, 15)
    rim = (70, 70, 70)
    riflesso = (200, 200, 200)

    # lenti piene
    cv2.circle(frame, (cx_l, cy), raggio, lente, -1)
    cv2.circle(frame, (cx_r, cy), raggio, lente, -1)
    # cornice
    cv2.circle(frame, (cx_l, cy), raggio, rim, 2)
    cv2.circle(frame, (cx_r, cy), raggio, rim, 2)
    # ponte centrale
    cv2.line(frame, (cx_l + raggio, cy), (cx_r - raggio, cy), rim, 3)
    # aste laterali
    cv2.line(frame, (cx_l - raggio, cy),
             (cx_l - raggio - int(raggio * 1.2), cy - raggio // 4), rim, 2)
    cv2.line(frame, (cx_r + raggio, cy),
             (cx_r + raggio + int(raggio * 1.2), cy - raggio // 4), rim, 2)
    # riflesso bianco sulla lente (effetto luce)
    r_high = max(2, raggio // 4)
    cv2.circle(frame, (cx_l - raggio // 3, cy - raggio // 3), r_high, riflesso, -1)
    cv2.circle(frame, (cx_r - raggio // 3, cy - raggio // 3), r_high, riflesso, -1)


def occhiali(frame, facce):
    # occhiali PNG agli occhi; se mancano disegna occhiali da sole rotondi
    path = os.path.join(ASSETS, "occhiali.png")
    for (fx, fy, fw, fh) in facce:
        roi_gray = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
        eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        if len(eyes) >= 2:
            # ordina gli occhi da sinistra a destra
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            gx = fx + ex1
            gy = fy + min(ey1, ey2) - int(eh1 * 0.3)
            gw = (ex2 + ew2) - ex1
            gh = int(max(eh1, eh2) * 1.6)
            if _sovrapponi_png(frame, path, gx, gy, gw, gh):
                continue
            cx_l = fx + ex1 + ew1 // 2
            cx_r = fx + ex2 + ew2 // 2
            cy = fy + ey1 + eh1 // 2
            raggio = max(ew1, ew2) // 2
            _disegna_occhiali(frame, cx_l, cx_r, cy, raggio)
        else:
            # nessun occhio rilevato → posiziono a stima sul terzo superiore
            cy = fy + int(fh * 0.42)
            raggio = int(fw * 0.16)
            cx_l = fx + int(fw * 0.28)
            cx_r = fx + int(fw * 0.72)
            _disegna_occhiali(frame, cx_l, cx_r, cy, raggio)
    return frame


# --- Baffi (handlebar) ----------------------------------------------------

def _disegna_baffi(frame, fx, fy, fw, fh):
    # baffi a manubrio centrati sotto al naso
    colore = (25, 18, 12)         # marrone scurissimo
    highlight = (50, 35, 25)      # accento più chiaro

    cx = fx + fw // 2
    cy = fy + int(fh * 0.74)      # appena sopra la bocca
    half_w = int(fw * 0.32)       # mezza larghezza
    half_h = max(4, int(fh * 0.05))  # mezza altezza

    # poligono baffi: due ali con dip centrale (philtrum)
    points = np.array([
        # parte alta (sotto il naso): dip al centro
        [cx - half_w,             cy - half_h // 2],
        [cx - int(half_w * 0.7),  cy - half_h],
        [cx - int(half_w * 0.3),  cy - half_h // 3],
        [cx,                       cy + half_h // 4],   # philtrum dip
        [cx + int(half_w * 0.3),  cy - half_h // 3],
        [cx + int(half_w * 0.7),  cy - half_h],
        [cx + half_w,             cy - half_h // 2],
        # punte arricciate verso l'alto
        [cx + half_w + half_h,    cy - int(half_h * 1.6)],
        [cx + int(half_w * 0.9),  cy + half_h],
        # parte bassa: ricurva verso il centro
        [cx + int(half_w * 0.4),  cy + int(half_h * 1.2)],
        [cx,                       cy + int(half_h * 0.7)],
        [cx - int(half_w * 0.4),  cy + int(half_h * 1.2)],
        [cx - int(half_w * 0.9),  cy + half_h],
        [cx - half_w - half_h,    cy - int(half_h * 1.6)],
    ], np.int32)
    cv2.fillPoly(frame, [points], colore)

    # piccolo highlight sopra per dare volume
    cv2.line(frame,
             (cx - int(half_w * 0.6), cy - int(half_h * 0.5)),
             (cx + int(half_w * 0.6), cy - int(half_h * 0.5)),
             highlight, 1, cv2.LINE_AA)


def baffi(frame, facce):
    # baffi PNG sopra al labbro; se mancano disegna un handlebar
    path = os.path.join(ASSETS, "baffi.png")
    for (fx, fy, fw, fh) in facce:
        bw = int(fw * 0.7)
        bh = int(fh * 0.18)
        bx = fx + (fw - bw) // 2
        by = fy + int(fh * 0.65)
        if not _sovrapponi_png(frame, path, bx, by, bw, bh):
            _disegna_baffi(frame, fx, fy, fw, fh)
    return frame


# --- Etichetta ------------------------------------------------------------

def etichetta(frame, facce):
    # scrive ETICHETTA sopra ogni faccia con sfondo nero semitrasparente
    for (fx, fy, fw, _) in facce:
        scala = fw / 200.0
        spessore = max(1, int(scala * 2))
        (tw, th), _ = cv2.getTextSize(ETICHETTA, cv2.FONT_HERSHEY_SIMPLEX,
                                       scala, spessore)
        tx = fx + (fw - tw) // 2
        ty = fy - 10
        cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, ETICHETTA, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scala, (0, 255, 255),
                    spessore, cv2.LINE_AA)
    return frame


# --- Charlie Kirk ---------------------------------------------------------
# Per ridurre il lag manteniamo in cache, per ogni dimensione (w, h):
#   - la faccia di Kirk già ridimensionata e convertita in float32
#   - la maschera ellittica sfumata (alpha)
#   - 1 - alpha (precalcolato per evitare di rifarlo ogni frame)
# La face detection in main.py gira 1 frame su 4, quindi per 4 frame
# consecutivi la dimensione della faccia è identica → 4 cache hit garantiti.

_charlie_face = None
_kirk_cache = {}     # (w, h) → (kirk_f32, alpha, 1-alpha)


def charlie_kirk_audio_start():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(ASSETS, "charlie_kirk.mp3"))
        pygame.mixer.music.play(-1)
    except Exception:
        pass


def charlie_kirk_audio_stop():
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass


def _kirk_assets(w, h):
    chiave = (w, h)
    if chiave in _kirk_cache:
        return _kirk_cache[chiave]
    kirk = cv2.resize(_charlie_face, (w, h)).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * 0.48), int(h * 0.48)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 12)
    alpha = mask[:, :, np.newaxis]
    _kirk_cache[chiave] = (kirk, alpha, 1.0 - alpha)
    return _kirk_cache[chiave]


def charlie_kirk(frame, facce):
    # sovrappone la faccia di Charlie Kirk con blend ellittico su ogni faccia
    global _charlie_face
    if _charlie_face is None:
        _charlie_face = cv2.imread(os.path.join(ASSETS, "charlie_face.png"))
    if _charlie_face is None or len(facce) == 0:
        return frame

    h_frame, w_frame = frame.shape[:2]
    for (fx, fy, fw, fh) in facce:
        x1, y1 = max(fx, 0), max(fy, 0)
        x2, y2 = min(fx + fw, w_frame), min(fy + fh, h_frame)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue
        kirk, alpha, inv_alpha = _kirk_assets(rw, rh)
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = (kirk * alpha + roi * inv_alpha).astype(np.uint8)
    return frame


# --- Effetto "67" (mano destra = 6, mano sinistra = 7) --------------------

import mediapipe as mp

_hand_detector = None
_67_ultimo_risultato = None
_67_frame_count = 0
_67_OGNI = 5            # esegue MediaPipe 1 frame su N (più alto = più fluido)
_67_INPUT_W = 240       # risoluzione fissa bassa per velocizzare il modello
_67_ts_ms = 0           # timestamp monotonico per la modalità VIDEO


def _get_hand_detector():
    # crea il detector una sola volta in modalità VIDEO (più veloce di IMAGE
    # perché internamente usa il tracking temporale fra frame consecutivi)
    global _hand_detector
    if _hand_detector is None:
        model_path = os.path.join(ASSETS, "hand_landmarker.task")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return _hand_detector


def prewarm_hand_detector():
    # chiamare all'avvio per evitare il lag al primo ingresso nel filtro 67
    _get_hand_detector()


def _disegna_numero(canvas, testo, cx, cy, size):
    # numero stile meme: bianco con bordo nero spesso, centrato in (cx, cy)
    font = cv2.FONT_HERSHEY_DUPLEX
    scala = size / 60.0
    bordo = max(2, int(scala * 12))
    fill = max(1, int(scala * 6))
    (tw, th), _ = cv2.getTextSize(testo, font, scala, bordo)
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(canvas, testo, (tx, ty), font, scala, (0, 0, 0), bordo, cv2.LINE_AA)
    cv2.putText(canvas, testo, (tx, ty), font, scala, (255, 255, 255), fill, cv2.LINE_AA)


def effetto_67(frame, facce):
    # mano destra → "6", mano sinistra → "7"
    global _67_ultimo_risultato, _67_frame_count, _67_ts_ms
    h, w = frame.shape[:2]
    size = int(min(h, w) * 0.25)

    _67_frame_count += 1
    if _67_frame_count % _67_OGNI == 0:
        # input molto piccolo (240px) per MediaPipe: detection veloce
        scala = _67_INPUT_W / w
        small_w = _67_INPUT_W
        small_h = int(h * scala)
        piccolo = cv2.resize(frame, (small_w, small_h))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(piccolo, cv2.COLOR_BGR2RGB))
        # timestamp monotonico richiesto dalla modalità VIDEO
        _67_ts_ms = max(_67_ts_ms + 33, int(time.time() * 1000))
        _67_ultimo_risultato = _get_hand_detector().detect_for_video(
            mp_image, _67_ts_ms)

    res = _67_ultimo_risultato
    if res and res.hand_landmarks and res.handedness:
        for lm, handedness in zip(res.hand_landmarks, res.handedness):
            polso = lm[0]
            cx = int(polso.x * w)
            cy = int(polso.y * h) - size // 2
            # NB: MediaPipe etichetta le mani dal punto di vista dell'immagine,
            # che con una webcam frontale risulta ribaltato rispetto all'utente.
            # Per questo "Left" → 6 e "Right" → 7 (invertito rispetto al label).
            numero = "6" if handedness[0].category_name == "Left" else "7"
            _disegna_numero(frame, numero, cx, cy, size)
    return frame


# ---------------------------------------------------------------------------
# Effetti di movimento (firma: frame, prev_frame_grezzo)
# ---------------------------------------------------------------------------

def rilevamento_movimento(frame, prev_frame):
    # tinge di rosso le zone diverse rispetto al frame precedente
    if prev_frame is None:
        return frame
    f1 = cv2.GaussianBlur(prev_frame, (21, 21), 0)
    f2 = cv2.GaussianBlur(frame, (21, 21), 0)
    diff = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    result = frame.copy()
    result[thresh == 255] = [0, 0, 220]
    return result


def ghost_effect(frame, prev_frame):
    # mescola 70% frame attuale + 30% precedente per creare una scia
    if prev_frame is None:
        return frame
    return cv2.addWeighted(frame, 0.7, prev_frame, 0.3, 0)
