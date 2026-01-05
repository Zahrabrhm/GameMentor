import os
import pickle
import time
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


def run_tutorial(weights_path, data_dir="data", frames_before_takeover=100):
    pygame.init()
    
    data_path = Path(data_dir)
    
    with open(data_path / "human_actions.pkl", "rb") as f:
        human_actions = pickle.load(f)
    
    with open(data_path / "mistakes.pkl", "rb") as f:
        mistakes = pickle.load(f)
    
    if not mistakes:
        print("No mistakes found! Great job!")
        return
    
    last_mistake = mistakes[-1]
    takeover_frame = max(0, last_mistake - frames_before_takeover)
    
    env = gym.make('LunarLander-v2')
    N_actions = env.action_space.n
    observation, info = env.reset(seed=123)
    N_state = len(observation)
    
    parameters = {'N_state': N_state, 'N_actions': N_actions}
    agent = DQN(parameters=parameters)
    agent.load_weights(weights_path)
    
    env = PyGameWrapper(gym.make('LunarLander-v2', render_mode='human'))
    
    state, info = env.reset(seed=123)
    done = False
    frame = 0
    agent_took_over = False
    
    print("\n=== Tutorial Mode ===")
    print(f"Replaying your actions until frame {takeover_frame}")
    print("Then the AI agent will demonstrate the correct approach...")
    
    while not done:
        for i in itertools.count():
            frame += 1
            
            if frame < takeover_frame:
                action = human_actions[i]
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                if not agent_took_over:
                    print("\n>>> AI Agent taking over! Watch and learn... <<<\n")
                    time.sleep(1)
                    agent_took_over = True
                
                action = agent.act(state, epsilon=0)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            state = next_state
            env.render()
            
            if done:
                if reward > 0:
                    print("Successfully landed!")
                else:
                    print("Crashed!")
                break
    
    env.close()
    print("\nTutorial complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tutorial showing correct gameplay")
    parser.add_argument("--weights", type=str, default="models/ddqn_weights.pth",
                        help="Path to trained agent weights")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing recorded gameplay data")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames before mistake to start agent takeover")
    args = parser.parse_args()
    
    run_tutorial(args.weights, args.data, args.frames)
