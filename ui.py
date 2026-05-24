import cv2
import numpy as np

COLORI_CATEGORIA = {
    "normal": (255, 200, 80),
    "face":   (120, 220, 120),
    "motion": (80, 160, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

def _pannello(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def _testo_centrato(frame, testo, cx, cy, scala, colore, spessore=1):
    (tw, th), _ = cv2.getTextSize(testo, FONT, scala, spessore)
    cv2.putText(frame, testo, (cx - tw // 2, cy + th // 2),
                FONT, scala, colore, spessore, cv2.LINE_AA)
    return tw, th

def _chip(frame, x, y, testo, colore, attivo=True):
    scala = 0.5
    spessore = 1
    (tw, th), _ = cv2.getTextSize(testo, FONT, scala, spessore)
    pad_x, pad_y = 10, 6
    w = tw + pad_x * 2
    h = th + pad_y * 2
    if attivo:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colore, -1)
        cv2.putText(frame, testo, (x + pad_x, y + pad_y + th),
                    FONT, scala, (20, 20, 20), spessore, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
        cv2.putText(frame, testo, (x + pad_x, y + pad_y + th),
                    FONT, scala, (180, 180, 180), spessore, cv2.LINE_AA)
    return w

def _barra_stato(frame, w, fps_val, n_facce, registrazione, auto_mode):
    _pannello(frame, 0, 0, w, 44, alpha=0.45)

    cv2.putText(frame, "FALCARI", (12, 28),
                FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "cam", (110, 28),
                FONT, 0.55, (120, 200, 255), 1, cv2.LINE_AA)

    x = w - 12
    chips = [
        (f"FPS {fps_val:4.1f}", (200, 200, 200), True),
        (f"FACES {n_facce}",    (120, 220, 120), True),
    ]
    if auto_mode:
        chips.append(("AUTO", (255, 200, 80), True))
    if registrazione:
        chips.append(("REC", (60, 60, 230), True))

    for testo, colore, attivo in chips:
        scala = 0.5
        (tw, _), _ = cv2.getTextSize(testo, FONT, scala, 1)
        chip_w = tw + 20
        x -= chip_w
        _chip(frame, x, 10, testo, colore, attivo)
        x -= 6

    if registrazione:
        t = cv2.getTickCount() / cv2.getTickFrequency()
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (w - 18, 22), 5, (0, 0, 230), -1)

def _pillola_filtro(frame, w, h, filtri, idx):
    nome, _fn, categoria = filtri[idx]
    colore_cat = COLORI_CATEGORIA.get(categoria, (200, 200, 200))

    pill_h = 56
    pill_w = min(560, w - 80)
    pill_x = (w - pill_w) // 2
    pill_y = h - pill_h - 56
    _pannello(frame, pill_x, pill_y, pill_w, pill_h, alpha=0.6)

    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                  colore_cat, 2)

    cx = w // 2
    cy = pill_y + pill_h // 2

    cv2.putText(frame, "<", (pill_x + 14, cy + 8),
                FONT, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, ">", (pill_x + pill_w - 28, cy + 8),
                FONT, 0.9, (180, 180, 180), 2, cv2.LINE_AA)

    nome_prev = filtri[(idx - 1) % len(filtri)][0]
    nome_next = filtri[(idx + 1) % len(filtri)][0]
    cv2.putText(frame, nome_prev, (pill_x + 36, cy + 6),
                FONT, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    (tw_next, _), _ = cv2.getTextSize(nome_next, FONT, 0.45, 1)
    cv2.putText(frame, nome_next, (pill_x + pill_w - 40 - tw_next, cy + 6),
                FONT, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    _testo_centrato(frame, nome, cx, cy - 4, 0.85, (255, 255, 255), 2)

    info = f"{idx + 1} / {len(filtri)}   {categoria.upper()}"
    _testo_centrato(frame, info, cx, cy + 18, 0.4, colore_cat, 1)

    bar_y = pill_y + pill_h + 6
    bar_w = pill_w
    bar_x = pill_x
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 3),
                  (60, 60, 60), -1)
    pos_w = int(bar_w * (idx + 1) / len(filtri))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + pos_w, bar_y + 3),
                  colore_cat, -1)

def _legenda_tasti(frame, w, h):
    _pannello(frame, 0, h - 26, w, 26, alpha=0.55)
    tasti = [
        ("A/D", "filtro"),
        ("1-9", "scelta"),
        ("S",   "foto"),
        ("R",   "rec"),
        ("P",   "auto"),
        ("Q",   "esci"),
    ]
    x = 12
    y = h - 8
    for tasto, descr in tasti:
        (tw, th), _ = cv2.getTextSize(tasto, FONT, 0.42, 1)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4),
                      (90, 90, 90), 1)
        cv2.putText(frame, tasto, (x, y),
                    FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        x += tw + 10
        cv2.putText(frame, descr, (x, y),
                    FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        (dw, _), _ = cv2.getTextSize(descr, FONT, 0.42, 1)
        x += dw + 18

def disegna_hud(frame, filtri, filtro_idx, nome_filtro, n_facce, fps_val,
                registrazione, auto_mode):
    h, w = frame.shape[:2]
    _barra_stato(frame, w, fps_val, n_facce, registrazione, auto_mode)
    _pillola_filtro(frame, w, h, filtri, filtro_idx)
    _legenda_tasti(frame, w, h)
    return frame
