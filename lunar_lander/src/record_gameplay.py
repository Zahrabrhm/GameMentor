import os
import pickle
import itertools
import numpy as np
import torch
import gymnasium as gym
import pygame
from pathlib import Path

from agent import DQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


class PyGameWrapper(gym.Wrapper):
    def render(self, **kwargs):
        retval = self.env.render(**kwargs)
        for event in pygame.event.get():
            pass
        return retval


def get_keyboard_action():
    try:
        import keyboard
        if keyboard.is_pressed("left"):
            return 1
        elif keyboard.is_pressed("right"):
            return 3
        elif keyboard.is_pressed("up"):
            return 2
        else:
            return 0
    except ImportError:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            return 1
        elif keys[pygame.K_RIGHT]:
            return 3
        elif keys[pygame.K_UP]:
            return 2
        else:
            return 0


def record_human_play(weights_path, output_dir="data"):
    pygame.init()
    
    env = gym.make('LunarLander-v2')
    N_actions = env.action_space.n
    observation, info = env.reset(seed=123)
    N_state = len(observation)
    
    parameters = {'N_state': N_state, 'N_actions': N_actions}
    agent = DQN(parameters=parameters)
    agent.load_weights(weights_path)
    
    env = PyGameWrapper(gym.make('LunarLander-v2', render_mode="human"))
    
    human_actions = []
    human_states = []
    agent_actions = []
    action_values_list = []
    
    state, info = env.reset(seed=123)
    episode_done = False
    total_reward = 0
    
    print("Use arrow keys to control the lander:")
    print("  UP: Fire main engine")
    print("  LEFT: Fire left engine")
    print("  RIGHT: Fire right engine")
    print("\nStarting game...")
    
    while not episode_done:
        for i in itertools.count():
            action = get_keyboard_action()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            human_actions.append(action)
            human_states.append(state)
            
            action_values = agent.act_values(state)
            action_values_list.append(action_values.tolist())
            
            agent_action = agent.act(state, epsilon=0)
            agent_actions.append(agent_action)
            
            total_reward += reward
            state = next_state
            env.render()
            
            if done:
                print(f'Episode finished! Steps: {i+1}, Total reward: {total_reward:.2f}')
                episode_done = True
                break
    
    env.close()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = {
        'human_actions': human_actions,
        'agent_actions': agent_actions,
        'action_values': action_values_list,
        'total_reward': total_reward
    }
    
    for name, value in data.items():
        filepath = Path(output_dir) / f"{name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
        print(f"Saved {name} to {filepath}")
    
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record human gameplay for Lunar Lander")
    parser.add_argument("--weights", type=str, default="models/ddqn_weights.pth",
                        help="Path to trained agent weights")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory for recorded data")
    args = parser.parse_args()
    
    record_human_play(args.weights, args.output)
