# 🎮 GameMentor: AI-Powered Personalized Game Tutorials

[![IEEE CIG 2024](https://img.shields.io/badge/IEEE%20CIG-2024-blue)](https://ieeexplore.ieee.org/document/10613541)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **GameMentor** is an AI-driven tutorial system that creates personalized learning experiences for video game players. Instead of one-size-fits-all tutorials, GameMentor analyzes your gameplay, identifies your mistakes, and generates customized tutorials that target your specific weaknesses.

## 🌐 Live Demo

**[Try the interactive demo →](https://Zahrabrhm.github.io/GameMentor/)**

Play Lunar Lander and Super Mario directly in your browser and experience the GameMentor system!

## 📄 Paper

This repository contains the implementation for our IEEE HSI 2024 paper:

**"GameMentor: Customized Tutorial for Video Games"**

📖 [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/10613541)

### Citation

```bibtex
@inproceedings{gamementor2024,
  title={GameMentor: Customized Tutorial for Video Games},
  author={[Authors]},
  booktitle={2024 IEEE Conference on Human System Interaction (HSI)},
  year={2024},
  organization={IEEE}
}
```

## 🎯 Overview

Video game tutorials are crucial for player onboarding, but traditional tutorials fail to account for individual skill variations. GameMentor solves this by:

1. **Training an Expert AI Agent** - Using Deep Reinforcement Learning to master the game
2. **Recording Human Gameplay** - Capturing player actions alongside AI recommendations
3. **Identifying Mistakes** - Detecting critical decision points where players made suboptimal choices
4. **Generating Personalized Tutorials** - Creating targeted practice scenarios based on individual weaknesses

![GameMentor Pipeline](docs/images/pipeline.png)

## 🚀 Features

- **🎮 Lunar Lander** - Classic control problem with Double Deep Q-Network agent
- **🍄 Super Mario Bros** - Platform game with CNN-based DQN agent
- **📊 Mistake Analysis** - Automated detection of critical gameplay errors
- **📚 Personalized Tutorials** - AI demonstrations of correct approach at mistake points
- **🏋️ Practice Mode** - Recreated scenarios for targeted skill improvement

## 📁 Repository Structure

```
GameMentor/
├── docs/                       # GitHub Pages web demo
│   ├── index.html
│   ├── styles.css
│   ├── lunar-lander.js
│   ├── super-mario.js
│   └── main.js
├── lunar_lander/               # Lunar Lander implementation
│   ├── src/
│   │   ├── agent.py           # DQN/Double DQN agent implementation
│   │   ├── train.py           # Agent training script
│   │   ├── record_gameplay.py # Human gameplay recording
│   │   ├── analyze_gameplay.py# Mistake detection
│   │   ├── tutorial.py        # Tutorial generation
│   │   └── practice.py        # Practice mode
│   ├── models/                # Pre-trained weights
│   └── data/                  # Recorded gameplay data
├── super_mario/               # Super Mario implementation
│   ├── src/
│   │   ├── mario_agent.py     # CNN-DQN agent and training
│   │   ├── record_gameplay.py
│   │   ├── analyze_gameplay.py
│   │   ├── tutorial.py
│   │   └── practice.py
│   ├── models/
│   └── data/
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GameMentor.git
cd GameMentor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Super Mario, also install:
pip install gym-super-mario-bros nes-py
```

## 📖 Usage

### Lunar Lander

#### 1. Train the Agent (or use pre-trained weights)

```bash
cd lunar_lander/src
python train.py --episodes 5000 --save-dir ../models
```

#### 2. Record Your Gameplay

```bash
python record_gameplay.py --weights ../models/ddqn_weights.pth --output ../data
```

Use arrow keys to control:
- ↑ : Fire main engine
- ← : Fire left thruster  
- → : Fire right thruster

#### 3. Analyze Your Performance

```bash
python analyze_gameplay.py --data ../data --threshold 12
```

#### 4. Watch the Tutorial

```bash
python tutorial.py --weights ../models/ddqn_weights.pth --data ../data
```

#### 5. Practice Mode

```bash
python practice.py --weights ../models/ddqn_weights.pth --data ../data
```

### Super Mario Bros

#### 1. Train the Agent

```bash
cd super_mario/src
python mario_agent.py --episodes 50000 --save-dir ../checkpoints
```

#### 2. Record Gameplay & Generate Tutorial

```bash
python record_gameplay.py --checkpoint ../checkpoints/mario_net_X.chkpt --output ../data
python analyze_gameplay.py --data ../data
python tutorial.py --checkpoint ../checkpoints/mario_net_X.chkpt --data ../data
```

## 🧠 Technical Details

### Lunar Lander Agent

| Component | Details |
|-----------|---------|
| Algorithm | Double Deep Q-Network (DDQN) |
| State Space | 8-dimensional continuous |
| Action Space | 4 discrete actions |
| Network | MLP: 8 → 128 → 64 → 4 |
| Training | ~2000-5000 episodes |

### Super Mario Agent

| Component | Details |
|-----------|---------|
| Algorithm | DQN with CNN |
| State Space | 84×84×4 grayscale frames |
| Action Space | 7 discrete actions |
| Network | 3 Conv layers + 2 Dense layers |
| Training | ~40000-50000 episodes |

### Mistake Detection

Mistakes are identified at **critical states** where:
- The Q-value difference between best and worst actions exceeds a threshold
- The human action differs from the AI-recommended action

This ensures we focus on moments where decisions actually matter.

## 📊 Results

Our user studies demonstrate significant improvements with GameMentor:

- **Faster skill acquisition** compared to traditional tutorials
- **Targeted improvement** in specific weak areas
- **Higher engagement** through personalized challenges
- **Better retention** of learned skills

See the [paper](https://ieeexplore.ieee.org/document/10613541) for detailed experimental results.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gym for the Lunar Lander environment
- Nintendo and gym-super-mario-bros for the Super Mario environment
- PyTorch team for the deep learning framework

## 📬 Contact

For questions about the paper or implementation, please open an issue or contact the authors.

---

<p align="center">
  <b>⭐ If you find this work useful, please consider giving it a star! ⭐</b>
</p>
