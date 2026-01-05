import os
import pickle
import torch
from pathlib import Path

from mario_agent import create_mario_env, MarioNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_keyboard_action():
    try:
        import keyboard
        if keyboard.is_pressed("6"):
            if keyboard.is_pressed("8"):
                return 4
            return 1
        elif keyboard.is_pressed("8"):
            return 5
        elif keyboard.is_pressed("4"):
            if keyboard.is_pressed("8"):
                return 3
            return 6
        else:
            return 0
    except ImportError:
        import pygame
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            if keys[pygame.K_UP]:
                return 4
            return 1
        elif keys[pygame.K_UP]:
            return 5
        elif keys[pygame.K_LEFT]:
            if keys[pygame.K_UP]:
                return 3
            return 6
        else:
            return 0


def record_human_play(checkpoint_path, output_dir="data"):
    state_dim = (4, 84, 84)
    action_dim = 7
    
    net = MarioNet(state_dim, action_dim).float()
    net = net.to(device=device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict['model'])
    
    env = create_mario_env(render_mode='human')
    
    human_actions = []
    human_states = []
    agent_actions = []
    action_values_list = []
    positions = []
    
    state = env.reset()
    episode_done = False
    
    print("Use numpad or arrow keys to control Mario:")
    print("  8/UP: Jump")
    print("  6/RIGHT: Move right")
    print("  4/LEFT: Move left")
    print("  8+6: Jump right")
    print("  8+4: Jump left")
    print("\nStarting game...")
    
    while not episode_done:
        action = get_keyboard_action()
        next_state, reward, done, trunc, info = env.step(action)
        
        human_actions.append(action)
        human_states.append(state)
        
        stateout = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        stateout = torch.tensor(stateout, device=device).unsqueeze(0)
        action_values = net(stateout, model="online")
        action_values_list.append(action_values.tolist())
        
        agent_action = torch.argmax(action_values[-1]).item()
        agent_actions.append(agent_action)
        
        positions.append(info["x_pos"])
        
        state = next_state
        env.render()
        
        if info["life"] < 2:
            print(f"Game over! Mario traveled to position: {positions[-1]}")
            episode_done = True
    
    env.close()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = {
        'human_actions': human_actions,
        'agent_actions': agent_actions,
        'action_values': action_values_list,
        'positions': positions
    }
    
    for name, value in data.items():
        filepath = Path(output_dir) / f"{name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
        print(f"Saved {name} to {filepath}")
    
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record human gameplay for Super Mario")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory for recorded data")
    args = parser.parse_args()
    
    record_human_play(args.checkpoint, args.output)
