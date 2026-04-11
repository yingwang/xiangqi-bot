# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + CNN vision + screen automation.

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 100% val accuracy) identifies pieces from screenshots
3. **FEN validation** enforces piece count rules and auto-corrects misclassifications using confidence scores
4. **Double-shot parsing** takes a second screenshot when confidence is low, averages probabilities to handle animation artifacts
5. **Quartz CGEvent** clicks to execute moves

```
Screenshot → CNN parse board → FEN validation → Compare with tracked state → Detect opponent move
                  ↓                                                                ↓
          (low confidence?)                              Pikafish best move ← Current FEN
                  ↓                                                                ↓
          2nd screenshot →                               CGEvent click → Execute move
          average probs
```

## Platform

**macOS only.** The bot uses macOS-specific APIs (Quartz CGEvent for clicking, `screencapture` for screenshots). The included Pikafish binary is compiled for macOS x86_64 (runs on Apple Silicon via Rosetta).

## Setup

### 1. Install Python dependencies

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

### 2. Install Pikafish engine

Pikafish is a Chinese chess engine (forked from Stockfish, rewritten for xiangqi rules). It's a C++ compiled binary that the bot communicates with via UCI protocol over stdin/stdout.

The repo includes a pre-built macOS x86_64 binary. If it doesn't work on your machine, build from source:

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon   # for Apple Silicon
# or: make -j profile-build ARCH=x86-64   # for Intel Mac
cp pikafish ../../pikafish                  # copy binary to project root
```

The NNUE file (`pikafish.nnue`) is already included — it's the neural network weights that Pikafish uses to evaluate positions. Pikafish looks for it in the working directory automatically.

### 3. Run

```bash
# Open 天天象棋 in WeChat, start a game, then:
python3 xiangqi_bot.py
```

On first run, follow the calibration prompts — move your mouse to 2 corners of the board so the bot knows where the grid is. Calibration is saved to `calib.json` and reused until you delete it.

Supports both red and black sides. Auto-detects orientation from king position.

## CNN Training

Training data is auto-collected during gameplay. Debug patches are saved to `debug/` organized by piece type for easy review.

### Data augmentation

The augmentation pipeline generates variants per original image:
- **Brightness** (0.8x, 1.2x)
- **Shift** (2px jitter in 4 directions)
- **Scale** (0.85x, 1.15x, 1.25x — simulates selected/enlarged pieces)
- **Low-resolution** (downscale to 20/24/30px then upscale — simulates small board blurriness)

### Improve accuracy

1. Play a game — debug patches auto-save to `debug/`
2. Review `debug/red_C/`, `debug/red_P/` etc. for misclassifications
3. Move wrong images to the correct `cnn_data/<piece_type>/` folder
4. Retrain:

```bash
python3 xiangqi_cnn.py augment   # regenerate augmented data
python3 xiangqi_cnn.py train --epochs 30
```

### Other CNN commands

```bash
python3 xiangqi_cnn.py collect   # Collect from initial position
python3 xiangqi_cnn.py augment   # Augment data (brightness, shift, scale, low-res simulation)
python3 xiangqi_cnn.py test      # Test on current board
```

## Continuous play (multi-game auto-restart)

`continuous_play.py` is a standalone CLI supervisor that wraps `Bot` in a
loop: when a game ends, it automatically clicks through the result popups
and starts the next game. Runs headless — no GUI dependency.

```bash
python3 continuous_play.py play                # unlimited games
python3 continuous_play.py play --max-games 5  # stop after 5 games
python3 continuous_play.py test-templates      # verify templates match current screen
python3 continuous_play.py detect-end          # one-shot end-of-game check
python3 continuous_play.py diag                # dump screenshot + window info
python3 continuous_play.py crop-templates      # print manual cropping workflow
```

**How it works:**

1. **Game-end detection** — strict (0.95) `cv2.matchTemplate` on a set of
   end-screen templates (`popup_end_banner`, `popup_level_up`,
   `popup_badge_earned`, `btn_play_again`, `btn_close_x_top`, `btn_confirm`,
   `btn_switch_opponent`). Also catches Pikafish `(none)` (mate/stalemate)
   and a 150s opponent-stuck safety net.
2. **Tiered recovery** — when a game ends, cycle through template clicks
   → blind relative-coord clicks → ESC×3 → refind-window until the board
   becomes visible again.
3. **Multi-scale template matching** — each template is matched at scales
   0.85–1.15x against a 0.35x downscaled screenshot for ~7x speedup over
   naive full-res matching. Stepped thresholds (0.92 → 0.88) with
   early-exit at 0.97+.
4. **Click verification** — every click takes before/after screenshots and
   uses `bot.images_changed` to confirm the screen reacted. Failed clicks
   go on a 30s cooldown to prevent death loops.
5. **Failure snapshots** — if recovery times out, a screenshot plus state
   log is dumped to `continuous_play_debug/<timestamp>/`. End-of-game
   screenshots are auto-saved to `end_snapshots/` for future template
   tuning.
6. **Ctrl+C** — graceful stop on first press; hard force-exit after 4s via
   a daemon-thread `os._exit(130)` watchdog, so a stuck subprocess can't
   block shutdown.

**Templates** live in `templates/`. Missing templates degrade gracefully
(tier 1 just has fewer candidates; tiers 2–4 still run).

**Does not modify** `xiangqi_bot.py` or `app.py` — both are still
available as the single-game entry points:

| Entry point | Multi-game? |
|---|---|
| `python3 xiangqi_bot.py` | no (single game) |
| `python3 app.py` / `天天象棋Bot.app` | no (GUI, single game) |
| `python3 continuous_play.py play` | **yes** (auto-restart) |

## macOS App

A native macOS GUI (`天天象棋Bot.app`) with board display, move history (四字记录法), and start/stop controls.

### Build

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

The app is output to `dist/天天象棋Bot.app`. It bundles Pikafish, the ONNX model, and all Python dependencies.

### Run directly (without building)

```bash
python3 app.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | macOS GUI app (AppKit/PyObjC, board view, move history) |
| `xiangqi_bot.py` | Bot engine (game loop, move execution, double-shot parsing) |
| `continuous_play.py` | Multi-game supervisor (auto-restart between games) |
| `xiangqi_cnn.py` | CNN model, training, inference, FEN validation |
| `xiangqi_cnn_onnx.py` | ONNX inference wrapper (no PyTorch needed) |
| `xiangqi_cnn.onnx` | Trained model weights (ONNX format) |
| `xiangqi_bot.spec` | PyInstaller build spec for macOS app |
| `hook-cv2.py` | PyInstaller runtime hook for OpenCV |
| `pikafish` | Pikafish engine binary (macOS x86_64) |
| `pikafish.nnue` | Neural network evaluation file for Pikafish |
| `cnn_data/` | Training data by piece type (`red_R/`, `black_r/`, `empty/`, etc.) |
| `debug/` | Debug patches from latest game session |
| `calib.json` | Grid calibration data |
| `templates/` | Popup button templates used by `continuous_play.py` |
| `end_snapshots/` | Auto-saved end-of-game screenshots (for template tuning) |
| `continuous_play_debug/` | Failure dumps from `continuous_play.py` (screenshot + state) |
