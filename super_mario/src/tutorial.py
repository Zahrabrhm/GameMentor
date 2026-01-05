import os
import pickle
import time
import torch
from pathlib import Path

from mario_agent import create_mario_env, MarioNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_tutorial(checkpoint_path, data_dir="data", frames_before_takeover=50):
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
    
    state_dim = (4, 84, 84)
    action_dim = 7
    
    net = MarioNet(state_dim, action_dim).float()
    net = net.to(device=device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict['model'])
    
    env = create_mario_env(render_mode='human')
    
    state = env.reset()
    frame = 0
    agent_took_over = False
    
    print("\n=== Tutorial Mode ===")
    print(f"Replaying your actions until frame {takeover_frame}")
    print("Then the AI agent will demonstrate the correct approach...")
    
    while True:
        frame += 1
        
        if frame < takeover_frame and frame < len(human_actions):
            action = human_actions[frame - 1]
        else:
            if not agent_took_over:
                print("\n>>> AI Agent taking over! Watch and learn... <<<\n")
                time.sleep(1)
                agent_took_over = True
            
            stateout = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            stateout = torch.tensor(stateout, device=device).unsqueeze(0)
            action_values = net(stateout, model="online")
            action = torch.argmax(action_values, axis=1).item()
        
        next_state, reward, done, trunc, info = env.step(action)
        state = next_state
        env.render()
        
        if done or info["flag_get"]:
            if info["flag_get"]:
                print("Level completed!")
            elif info["life"] < 2:
                print("Mario died!")
            break
    
    env.close()
    print("\nTutorial complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run tutorial showing correct gameplay")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing recorded gameplay data")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames before mistake to start agent takeover")
    args = parser.parse_args()
    
    run_tutorial(args.checkpoint, args.data, args.frames)
