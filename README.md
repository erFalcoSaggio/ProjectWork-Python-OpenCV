# Falcari Cam

App Python che apre la webcam e applica in tempo reale filtri OpenCV, overlay
sulle facce (cappello, occhiali, barba, baffi, etichetta) ed effetti di
movimento. Tutto si controlla da tastiera. Codice diviso in moduli:
`main.py`, `filters.py`, `effects.py`, `ui.py` + cartella `assets/`.

## Requisiti

- Linux / macOS / Windows
- Python **3.10 o superiore** (testato con 3.12)
- Una webcam funzionante (USB o integrata)
- ~250 MB di spazio disco per dipendenze e asset

## Installazione (da zero)

```bash
git clone https://github.com/erFalcoSaggio/ProjectWork-Python-OpenCV.git
cd ProjectWork-Python-OpenCV
chmod +x run.sh
./run.sh
```

`run.sh` al **primo lancio** crea un virtualenv in `venv/`, installa le
dipendenze elencate in `requirements.txt` e avvia l'app. Ai lanci successivi
riusa direttamente il virtualenv.

Se preferisci fare tutto a mano:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Come avviare

```bash
./run.sh
```

## Tasti

| Tasto         | Azione                              |
|---------------|-------------------------------------|
| `A` / `←`     | filtro precedente                   |
| `D` / `→`     | filtro successivo                   |
| `1`-`9`, `0`  | scelta rapida del filtro            |
| `S`           | screenshot (salvato con timestamp)  |
| `R`           | avvia/ferma registrazione `.mp4`    |
| `P`           | modalità automatica on/off          |
| `Q`           | esci                                |

## Filtri disponibili

- **Colore**: Normale, Grigio, Negativo, Sepia, Heatmap, Cartoon, Pixelate,
  Vignetta, Motion Blur, Flip specchio, Glitch, Esplosione, Disco, Shake,
  Statica TV, Zoom Pulse
- **Sul viso** (face detection): Sfondo sfocato, Cappello, Occhiali, Baffi,
  Barba, Etichetta, Charlie Kirk (con audio in loop), 67 (mano destra → "6",
  mano sinistra → "7", richiede MediaPipe)
- **Movimento**: Rilevamento movimento, Ghost effect

## Struttura del progetto

```
ProjectWork-Python-OpenCV/
├── main.py            # loop principale, gestione tasti
├── filters.py         # filtri colore puri (frame in → frame out)
├── effects.py         # face detection, overlay PNG, motion
├── ui.py              # HUD (barra status, pillola filtro, legenda tasti)
├── requirements.txt
├── run.sh
├── README.md
└── assets/
    ├── cappello.png       # PNG con canale alpha per gli overlay
    ├── occhiali.png
    ├── baffi.png
    ├── barba.png
    ├── charlie_face.png
    ├── charlie_kirk.mp3   # audio in loop per il filtro Charlie Kirk
    ├── hand_landmarker.task  # modello MediaPipe per il filtro 67
    └── 6.png / 7.png / 67.png
```

## Note per Raspberry Pi

- Testato su Raspberry Pi 4 (4 GB) con Raspberry Pi OS 64-bit.
- Sul Pi **abbassa la risoluzione** della webcam o tieni alto `FACE_DETECT_OGNI`
  in `main.py` (di default già a 4) per mantenere il framerate ragionevole.
- **MediaPipe** (filtro "67") richiede la build ARM 64-bit. Su Raspberry Pi OS
  64-bit:
  ```bash
  pip install mediapipe
  ```
  Su sistemi 32-bit MediaPipe non è disponibile: il filtro 67 non si attiva
  ma tutti gli altri filtri continuano a funzionare.
- Per la webcam USB classe UVC non serve nulla. Per il modulo CSI usa
  `libcamera` + `v4l2loopback` (fuori dallo scopo di questo README).
- Se compare l'errore *"Cannot open camera"* verifica con `v4l2-ctl --list-devices`
  che la webcam sia visibile e che l'utente sia nel gruppo `video`.
