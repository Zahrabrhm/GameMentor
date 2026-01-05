# Lunar Lander Data

This directory stores gameplay recordings and analysis results.

## Contents

- `gameplay_*.pkl` - Recorded human gameplay sessions
- `mistakes_*.json` - Detected mistakes from gameplay analysis

## Recording Gameplay

```bash
cd ../src
python record_gameplay.py --output ../data/gameplay_session.pkl
```

## Analyzing Gameplay

```bash
python analyze_gameplay.py --recording ../data/gameplay_session.pkl --output ../data/mistakes.json
```
