import argparse
import datetime
from pathlib import Path

import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_agent import Mario, MetricLogger, SkipFrame, GrayScaleObservation, ResizeObservation


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = torch.tensor_wrapper(env)
    return env


class TensorWrapper:
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        state, info = self.env.reset()
        return torch.tensor(state.__array__(), dtype=torch.float32).unsqueeze(0), info
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        return (
            torch.tensor(next_state.__array__(), dtype=torch.float32).unsqueeze(0),
            reward,
            done,
            truncated,
            info
        )
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def torch_tensor_wrapper(env):
    return TensorWrapper(env)


def train(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = save_dir / "mario_net.chkpt"
    
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", 
        render_mode="rgb_array", 
        apply_api_compatibility=True
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir
    )
    
    if args.load_checkpoint and checkpoint_path.exists():
        mario.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    logger = MetricLogger(save_dir)
    
    episodes = args.episodes
    
    for e in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state.__array__(), dtype=torch.float32).unsqueeze(0)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = mario.act(state)
            
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state.__array__(), dtype=torch.float32).unsqueeze(0)
            done = done or truncated
            
            mario.cache(state, next_state, action, reward, done)
            
            q, loss = mario.learn()
            
            logger.log_step(reward, loss, q)
            
            state = next_state
            episode_reward += reward
        
        logger.log_episode()
        
        if (e + 1) % args.log_interval == 0:
            logger.record(episode=e + 1, epsilon=mario.exploration_rate, step=mario.curr_step)
        
        if (e + 1) % args.save_interval == 0:
            mario.save()
            print(f"Episode {e + 1}: Saved checkpoint")
    
    env.close()
    print("Training complete!")


def evaluate(args):
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", 
        render_mode="human", 
        apply_api_compatibility=True
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=Path(args.save_dir)
    )
    
    checkpoint_path = Path(args.save_dir) / "mario_net.chkpt"
    if checkpoint_path.exists():
        mario.load(checkpoint_path)
        mario.exploration_rate = 0.0
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found, using untrained agent")
    
    total_rewards = []
    
    for e in range(args.eval_episodes):
        state, _ = env.reset()
        state = torch.tensor(state.__array__(), dtype=torch.float32).unsqueeze(0)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = mario.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state.__array__(), dtype=torch.float32).unsqueeze(0)
            done = done or truncated
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {e + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(total_rewards):.2f}")
    print(f"  Std Reward: {np.std(total_rewards):.2f}")
    print(f"  Max Reward: {np.max(total_rewards):.2f}")
    print(f"  Min Reward: {np.min(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Super Mario agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="../models",
                        help="Directory to save/load model")
    parser.add_argument("--load-checkpoint", action="store_true",
                        help="Load existing checkpoint before training")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Episodes between logging")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Episodes between saving checkpoints")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
