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


def run_practice(weights_path, data_dir="data", frames_before_takeover=100):
    pygame.init()
    
    data_path = Path(data_dir)
    
    with open(data_path / "human_actions.pkl", "rb") as f:
        human_actions = pickle.load(f)
    
    with open(data_path / "mistakes.pkl", "rb") as f:
        mistakes = pickle.load(f)
    
    if not mistakes:
        print("No mistakes found! Play a new game.")
        return
    
    last_mistake = mistakes[-1]
    practice_start_frame = max(0, last_mistake - frames_before_takeover)
    
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
    practice_mode = False
    waiting_for_input = True
    total_reward = 0
    
    print("\n=== Practice Mode ===")
    print(f"Replaying your actions until frame {practice_start_frame}")
    print("Then YOU take control to practice the correct approach!")
    print("\nControls:")
    print("  UP: Fire main engine")
    print("  LEFT: Fire left engine")
    print("  RIGHT: Fire right engine")
    
    while not done:
        for i in itertools.count():
            frame += 1
            
            if frame < practice_start_frame:
                action = human_actions[i]
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                if not practice_mode:
                    print("\n>>> YOUR TURN! Press any arrow key to start... <<<\n")
                    practice_mode = True
                
                if waiting_for_input:
                    action = get_keyboard_action()
                    if action != 0:
                        waiting_for_input = False
                        time.sleep(0.5)
                
                if not waiting_for_input:
                    action = get_keyboard_action()
                else:
                    action = 0
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            state = next_state
            env.render()
            
            if done:
                print(f"\nPractice complete! Your reward: {total_reward:.2f}")
                if total_reward > 100:
                    print("Great improvement!")
                elif total_reward > 0:
                    print("Good try! Keep practicing.")
                else:
                    print("Keep practicing - you'll get better!")
                break
    
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Practice mode for Lunar Lander")
    parser.add_argument("--weights", type=str, default="models/ddqn_weights.pth",
                        help="Path to trained agent weights")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing recorded gameplay data")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames before mistake to start practice")
    args = parser.parse_args()
    
    run_practice(args.weights, args.data, args.frames)
