import pickle
import numpy as np
from pathlib import Path


def analyze_gameplay(data_dir="data", threshold=12):
    data_path = Path(data_dir)
    
    with open(data_path / "human_actions.pkl", "rb") as f:
        human_actions = pickle.load(f)
    
    with open(data_path / "agent_actions.pkl", "rb") as f:
        agent_actions = pickle.load(f)
    
    with open(data_path / "action_values.pkl", "rb") as f:
        action_values = pickle.load(f)
    
    with open(data_path / "total_reward.pkl", "rb") as f:
        total_reward = pickle.load(f)
    
    important_states = []
    mistakes = []
    correct_actions = []
    
    for i in range(len(action_values)):
        values = action_values[i]
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], list):
                values = values[0]
        
        value_range = abs(max(values) - min(values))
        
        if value_range > threshold:
            important_states.append(values[0] if isinstance(values[0], (int, float)) else i)
            
            if agent_actions[i] == human_actions[i]:
                print(f"State {i}: Human played correctly")
                correct_actions.append(i)
            else:
                print(f"State {i}: Human made a mistake")
                mistakes.append(i)
        else:
            important_states.append(0)
    
    total_important = len(correct_actions) + len(mistakes)
    
    if total_important > 0:
        correct_rate = (len(correct_actions) / total_important) * 100
        mistake_rate = (len(mistakes) / total_important) * 100
    else:
        correct_rate = 0
        mistake_rate = 0
    
    print(f"\n=== Gameplay Analysis ===")
    print(f"Total important states: {total_important}")
    print(f"Correct actions: {len(correct_actions)} ({correct_rate:.1f}%)")
    print(f"Mistakes: {len(mistakes)} ({mistake_rate:.1f}%)")
    print(f"Total reward: {total_reward:.2f}")
    
    results = {
        'mistakes': mistakes,
        'correct_actions': correct_actions,
        'important_states': important_states,
        'mistake_rate': mistake_rate,
        'correct_rate': correct_rate
    }
    
    for name, value in results.items():
        filepath = data_path / f"{name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze gameplay and identify mistakes")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory containing recorded gameplay data")
    parser.add_argument("--threshold", type=float, default=12,
                        help="Q-value difference threshold for important states")
    args = parser.parse_args()
    
    analyze_gameplay(args.data, args.threshold)
