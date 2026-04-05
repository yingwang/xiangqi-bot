# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + CNN vision + screen automation.

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 99.8% val accuracy) identifies pieces from screenshots
3. **FEN validation** enforces piece count rules and auto-corrects misclassifications using confidence scores
4. **Double-shot parsing** takes a second screenshot when confidence is low, averages probabilities to handle animation artifacts
5. **Quartz CGEvent** clicks to execute moves

## Architecture

```
Screenshot → CNN parse board → FEN validation → Compare with tracked state → Detect opponent move
                  ↓                                                                ↓
          (low confidence?)                              Pikafish best move ← Current FEN
                  ↓                                                                ↓
          2nd screenshot →                               CGEvent click → Execute move
          average probs
```

## Requirements

- macOS with WeChat (天天象棋 mini program)
- Python 3 with `opencv-python`, `numpy`, `pyautogui`, `torch`, `torchvision`
- Pikafish binary + `pikafish.nnue` in the project directory

## Setup

1. Place `pikafish` binary and `pikafish.nnue` in the project root
2. Open 天天象棋 in WeChat, start a game
3. Run: `python3 xiangqi_bot.py`
4. First run: follow calibration prompts to set grid coordinates

## Usage

```bash
python3 xiangqi_bot.py
```

Supports both red and black sides. Auto-detects orientation from king position.

## CNN Training

Training data is auto-collected during gameplay. Debug patches are saved to `debug/` organized by piece type for easy review.

### Workflow: improve accuracy

1. Play a game — debug patches auto-save to `debug/`
2. Review `debug/red_C/`, `debug/red_P/` etc. for misclassifications
3. Move wrong images to the correct `cnn_data/<piece_type>/` folder
4. Retrain:

```bash
python3 xiangqi_cnn.py train --epochs 30
```

### Other CNN commands

```bash
# Collect from initial position
python3 xiangqi_cnn.py collect

# Augment data (brightness, shift, rotation, scale)
python3 xiangqi_cnn.py augment

# Test on current board
python3 xiangqi_cnn.py test
```

## Files

- `xiangqi_bot.py` - Main bot script (game loop, move execution, double-shot parsing)
- `xiangqi_cnn.py` - CNN model, training, inference, and FEN validation
- `xiangqi_cnn.pt` - Trained model weights
- `pikafish` - Pikafish engine binary
- `pikafish.nnue` - Pikafish neural network evaluation file
- `cnn_data/` - Training data (organized by piece type: `red_R/`, `black_r/`, `empty/`, etc.)
- `debug/` - Debug patches from latest game session (same folder structure as `cnn_data/`)
- `calib.json` - Grid calibration data
