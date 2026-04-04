# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + screen automation.

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **screencapture** reads the board from the WeChat mini program window
3. **Quartz CGEvent** clicks to execute moves
4. **Perft enumeration** detects opponent moves by scoring pixel changes against all legal moves

## Requirements

- macOS with WeChat (天天象棋 mini program)
- Python 3 with `opencv-python`, `numpy`, `pyautogui`
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

## Status

Work in progress. The bot can play 10-20+ moves automatically with Pikafish-level strength. Opponent move detection is ~80% accurate; captures and fast opponent responses can cause detection failures.
