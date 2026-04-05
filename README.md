# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + CNN vision + screen automation.

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 100% accuracy) identifies pieces on the board from a single screenshot
3. **screencapture** reads the board from the WeChat mini program window
4. **Quartz CGEvent** clicks to execute moves
5. **Move detection** compares CNN-parsed board state against tracked state to find opponent's move

## Architecture

```
Screenshot → CNN parse board → Compare with tracked state → Detect opponent move
                                                          ↓
                              Pikafish best move ← Current FEN
                                                          ↓
                              CGEvent click → Execute move
```

Detection priority: CNN > pixel diff > occupancy analysis > wait for opponent

## Requirements

- macOS with WeChat (天天象棋 mini program)
- Python 3 with `opencv-python`, `numpy`, `pyautogui`, `torch`, `torchvision`
- [Pikafish](https://github.com/official-pikafish/Pikafish) compiled locally

## Setup

1. Compile Pikafish and place at `/tmp/pikafish-src/src/pikafish`
2. Open 天天象棋 in WeChat, start a game
3. Run calibration (first time): `python3 xiangqi_bot.py`
4. Follow prompts to set grid coordinates

## Usage

```bash
python3 xiangqi_bot.py
```

Supports both red and black sides. When playing black, the bot waits for the opponent's first move before starting.

## CNN Training

Training data is auto-collected during gameplay. To retrain:

```bash
# Collect from initial position
python3 xiangqi_cnn.py collect

# Augment data (brightness, shift, rotation)
python3 xiangqi_cnn.py augment

# Train (uses MPS on Apple Silicon)
python3 xiangqi_cnn.py train --epochs 40

# Test on current board
python3 xiangqi_cnn.py test
```

## Files

- `xiangqi_bot.py` - Main bot script
- `xiangqi_cnn.py` - CNN model, training, and inference
- `xiangqi_cnn.pt` - Trained model weights
- `cnn_data/` - Training data (organized by piece type: `red_R/`, `black_r/`, `empty/`, etc.)
- `calib.json` - Grid calibration data
