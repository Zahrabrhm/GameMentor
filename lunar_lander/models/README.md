# Lunar Lander Models

Place your trained model weights here.

## Expected Files

- `dqn_weights.pth` - DQN agent weights
- `actor_critic_weights.pth` - Actor-Critic agent weights (optional)

## Training

To train a new model:

```bash
cd ../src
python train.py --mode train --episodes 2000
```

The trained weights will be saved to this directory.
