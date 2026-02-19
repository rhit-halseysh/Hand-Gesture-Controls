# Hand-Gesture-Controls

Control YouTube Shorts with hand gestures using your webcam.

---

## Requirements

- Python 3.10+
- A webcam
- The [CSSE463 Doom Scroll Chrome Extension](https://chromewebstore.google.com/detail/csse463-doom-scroll-exten/ciciaichjicddbmhfojlgjdjjohddlpd) installed in Chrome

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv hand
```

**Windows (PowerShell):**
```powershell
.\hand\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
hand\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source hand/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python main.py
```

A window will open showing your webcam feed. Click **Start Tracking** in the UI to begin gesture detection.

---

## Keyboard Controls

| Key     | Action                        |
|---------|-------------------------------|
| `m`     | Toggle hand-as-mouse on/off   |
| `i`     | Toggle hand landmark overlay  |
| `q`     | Quit                          |
| `g`     | Toggle between Gesture/ASL mode |

---

## Gesture Actions

The extension must be installed and active on a YouTube Shorts page for these to work.

### Core Gestures

| Gesture         | Action            |
|-----------------|-------------------|
| Peace ✌️         | Next short        |
| Inverted Peace / Palm | Previous short |
| Thumbs Up / Like / Hand Heart | Like video |
| Dislike         | Dislike video     |
| Call 🤙          | Open comments     |
| Mute   | Close comments    |
| Fist ✊          | Pause / Play      |

### All Gesture Mappings

The full list of registered gestures and their actions is in [src/action_handler.py](src/action_handler.py) inside `_register_youtube_actions()`.

---

## Hand as Mouse

Press `m` to enable mouse control. When active, your palm position controls the cursor.

Additional mouse gestures (only active when mouse control is enabled):

| Gesture       | Action       |
|---------------|--------------|
| One finger ☝️  | Left click   |
| Little finger | Right click  |
| Point         | Toggle drag  |

---

## Landmark Overlay

Press `i` to toggle the hand landmark skeleton drawn over the webcam feed.

## ASL Mode

Press `g` to switch between normal gesture mode and ASL recognition mode. In ASL mode, the model will attempt to recognize American Sign Language letters instead of the core gesture set.