import os
import pickle
import time
import torch
from pathlib import Path

from mario_agent import create_mario_env, MarioNet
from record_gameplay import get_keyboard_action

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_practice(checkpoint_path, data_dir="data", frames_before_takeover=50):
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
    
    env = create_mario_env(render_mode='human')
    
    state = env.reset()
    frame = 0
    practice_mode = False
    waiting_for_input = True
    
    print("\n=== Practice Mode ===")
    print(f"Replaying your actions until frame {practice_start_frame}")
    print("Then YOU take control to practice the correct approach!")
    print("\nControls:")
    print("  8/UP: Jump")
    print("  6/RIGHT: Move right")
    print("  4/LEFT: Move left")
    print("  8+6: Jump right")
    print("  8+4: Jump left")
    
    while True:
        frame += 1
        
        if frame < practice_start_frame and frame < len(human_actions):
            action = human_actions[frame - 1]
        else:
            if not practice_mode:
                print("\n>>> YOUR TURN! Press any key to start... <<<\n")
                practice_mode = True
            
            if waiting_for_input:
                action = get_keyboard_action()
                if action != 0:
                    waiting_for_input = False
                    time.sleep(0.3)
            
            if not waiting_for_input:
                action = get_keyboard_action()
            else:
                action = 0
        
        next_state, reward, done, trunc, info = env.step(action)
        state = next_state
        env.render()
        
        if done or info["flag_get"]:
            if info["flag_get"]:
                print("\nCongratulations! Level completed!")
            elif info["life"] < 2:
                print(f"\nMario died at position {info['x_pos']}. Keep practicing!")
            break
    
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Practice mode for Super Mario")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing recorded gameplay data")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames before mistake to start practice")
    args = parser.parse_args()
    
    run_practice(args.checkpoint, args.data, args.frames)
