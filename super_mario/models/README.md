# Super Mario Models

Place your trained model weights here.

## Expected Files

- `mario_net.chkpt` - CNN-DQN agent checkpoint

## Training

To train a new model:

```bash
cd ../src
python train.py --mode train --episodes 50000
```

Training typically requires 40,000-50,000 episodes for good performance.

The trained weights will be saved to this directory.
