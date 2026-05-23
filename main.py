# Falcari - loop principale: legge la webcam, applica il filtro corrente,
# disegna l'interfaccia e gestisce i tasti.
import cv2
import time
from datetime import datetime

import filters
import effects
from ui import disegna_hud

# ---------------------------------------------------------------------------
# Catalogo dei filtri.
# Ogni voce: (nome mostrato, funzione, categoria).
# La categoria decide quali parametri extra la funzione riceve:
#   "normal" → solo frame                                  fn(frame)
#   "face"   → frame + lista di facce rilevate             fn(frame, facce)
#   "motion" → frame + frame precedente (grezzo)           fn(frame, prev)
# ---------------------------------------------------------------------------
FILTRI = [
    ("Normale",        filters.normale,               "normal"),
    ("Grigio",         filters.grigio,                "normal"),
    ("Negativo",       filters.negativo,              "normal"),
    ("Sepia",          filters.sepia,                 "normal"),
    ("Heatmap",        filters.heatmap,               "normal"),
    ("Cartoon",        filters.cartoon,               "normal"),
    ("Pixelate",       filters.pixelate,              "normal"),
    ("Vignetta",       filters.vignetta,              "normal"),
    ("Motion Blur",    filters.motion_blur,           "normal"),
    ("Flip specchio",  filters.flip_specchio,         "normal"),
    ("Glitch",         filters.glitch,                "normal"),
    ("Esplosione",     filters.esplosione,            "normal"),
    ("Disco",          filters.disco,                 "normal"),
    ("Shake",          filters.shake,                 "normal"),
    ("Statica TV",     filters.statica_tv,            "normal"),
    ("Zoom Pulse",     filters.zoom_pulse,            "normal"),
    ("Sfondo sfocato", effects.sfondo_sfocato,        "face"),
    ("Cappello",       effects.cappello,              "face"),
    ("Occhiali",       effects.occhiali,              "face"),
    ("Baffi",          effects.baffi,                 "face"),
    ("Etichetta",      effects.etichetta,             "face"),
    ("Charlie Kirk",   effects.charlie_kirk,          "face"),
    ("67",             effects.effetto_67,            "face"),
    ("Mov. detect",    effects.rilevamento_movimento, "motion"),
    ("Ghost",          effects.ghost_effect,          "motion"),
]

NOME_CHARLIE = "Charlie Kirk"
AUTO_INTERVALLO = 3   # secondi tra un filtro e l'altro in modalità AUTO

# ---------------------------------------------------------------------------
# Face detection: lavoriamo su un frame ridotto e solo 1 frame su N per
# tenere alti gli FPS. Le coordinate vengono poi riscalate.
# ---------------------------------------------------------------------------
FACE_DETECT_OGNI = 4
FACE_SCALE = 0.5

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def rileva_facce(frame):
    h, w = frame.shape[:2]
    piccolo = cv2.resize(frame, (int(w * FACE_SCALE), int(h * FACE_SCALE)))
    gray = cv2.cvtColor(piccolo, cv2.COLOR_BGR2GRAY)
    facce = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(facce) == 0:
        return facce
    # riporta le coordinate al frame originale
    return (facce / FACE_SCALE).astype(int)


def applica_filtro(frame, prev_raw, facce, idx):
    # esegue la funzione del filtro in base alla sua categoria
    fn = FILTRI[idx][1]
    categoria = FILTRI[idx][2]
    if categoria == "face":
        return fn(frame, facce)
    if categoria == "motion":
        return fn(frame, prev_raw)
    return fn(frame)


def gestisci_audio_charlie(nome_corrente, nome_precedente):
    # avvia o ferma l'audio di Charlie Kirk quando si entra/esce dal filtro
    if nome_corrente == nome_precedente:
        return
    if nome_corrente == NOME_CHARLIE:
        effects.charlie_kirk_audio_start()
    elif nome_precedente == NOME_CHARLIE:
        effects.charlie_kirk_audio_stop()


def avvia_registrazione(frame):
    # crea un VideoWriter con timestamp; restituisce (writer, nome_file)
    h, w = frame.shape[:2]
    nome_file = datetime.now().strftime("rec_%Y%m%d_%H%M%S.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(nome_file, fourcc, 20.0, (w, h))
    return writer, nome_file


def main():
    # --- stato del loop ---
    filtro_idx = 0
    filtro_precedente = ""       # per gestire l'audio di Charlie Kirk
    auto_mode = False
    auto_last = time.time()
    fps = 0.0
    fps_prev = time.time()
    prev_frame_raw = None        # frame grezzo precedente (per filtri "motion")
    facce = []
    frame_count = 0
    registrazione = False
    video_writer = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam")
        return

    # pre-carica MediaPipe (filtro "67") in modo che il primo ingresso nel
    # filtro non blocchi il loop video con il caricamento del modello
    print("Carico il modello hand-tracking...")
    effects.prewarm_hand_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # salva subito il frame grezzo: ci servirà al prossimo ciclo per i
        # filtri di movimento (devono confrontare due frame originali)
        frame_raw = frame.copy()

        # FPS istantaneo
        now = time.time()
        fps = 1.0 / (now - fps_prev + 1e-9)
        fps_prev = now

        # modalità AUTO: cambia filtro ogni AUTO_INTERVALLO secondi
        if auto_mode and (now - auto_last) >= AUTO_INTERVALLO:
            filtro_idx = (filtro_idx + 1) % len(FILTRI)
            auto_last = now

        # face detection ogni FACE_DETECT_OGNI frame (per non saturare la CPU)
        frame_count += 1
        if frame_count % FACE_DETECT_OGNI == 0:
            facce = rileva_facce(frame)

        nome_filtro = FILTRI[filtro_idx][0]
        gestisci_audio_charlie(nome_filtro, filtro_precedente)
        filtro_precedente = nome_filtro

        frame = applica_filtro(frame, prev_frame_raw, facce, filtro_idx)
        prev_frame_raw = frame_raw

        frame = disegna_hud(frame, FILTRI, filtro_idx, nome_filtro,
                            len(facce), fps, registrazione, auto_mode)

        if registrazione and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Falcari Cam", frame)

        # --- gestione tasti ---
        tasto = cv2.waitKey(1) & 0xFF

        if tasto == ord('q'):
            break

        elif tasto == ord('s'):
            nome_file = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(nome_file, frame)
            print(f"Screenshot salvato: {nome_file}")

        elif tasto == ord('r'):
            if not registrazione:
                video_writer, nome_file = avvia_registrazione(frame)
                registrazione = True
                print(f"Registrazione avviata: {nome_file}")
            else:
                registrazione = False
                video_writer.release()
                video_writer = None
                print("Registrazione fermata")

        elif tasto == ord('p'):
            auto_mode = not auto_mode
            auto_last = now
            print(f"Modalità auto: {'ON' if auto_mode else 'OFF'}")

        elif tasto in (81, ord('a')):   # freccia sinistra / A
            filtro_idx = (filtro_idx - 1) % len(FILTRI)

        elif tasto in (83, ord('d')):   # freccia destra / D
            filtro_idx = (filtro_idx + 1) % len(FILTRI)

        elif ord('1') <= tasto <= ord('9'):
            idx = tasto - ord('1')
            if idx < len(FILTRI):
                filtro_idx = idx

        elif tasto == ord('0') and len(FILTRI) >= 10:
            filtro_idx = 9

    # cleanup all'uscita
    effects.charlie_kirk_audio_stop()
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
