import os
import argparse
import gymnasium as gym
import torch
from pathlib import Path

from agent import DQN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_agent(
    n_episodes=5000,
    save_dir="models",
    solving_threshold_mean=230,
    solving_threshold_min=200
):
    env = gym.make('LunarLander-v2')
    N_actions = env.action_space.n
    observation, info = env.reset(seed=42)
    N_state = len(observation)
    
    parameters = {
        'N_state': N_state,
        'N_actions': N_actions,
        'n_episodes_max': n_episodes,
        'solving_threshold_mean': solving_threshold_mean,
        'solving_threshold_min': solving_threshold_min,
        'doubledqn': True,
        'neural_networks': {
            'policy_net': {'layers': [N_state, 128, 64, N_actions]},
            'target_net': {'layers': [N_state, 128, 64, N_actions]}
        }
    }
    
    agent = DQN(parameters=parameters)
    
    print("=" * 50)
    print("Training Double DQN Agent for Lunar Lander")
    print("=" * 50)
    print(f"State space: {N_state}")
    print(f"Action space: {N_actions}")
    print(f"Device: {device}")
    print(f"Max episodes: {n_episodes}")
    print("=" * 50)
    
    results = agent.train(
        environment=env,
        verbose=True,
        model_filename=None
    )
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    weights_path = Path(save_dir) / "ddqn_weights.pth"
    agent.save_weights(str(weights_path))
    
    print("\n" + "=" * 50)
    print(f"Training complete!")
    print(f"Final mean reward: {sum(results['episode_returns'][-20:]) / 20:.2f}")
    print(f"Weights saved to: {weights_path}")
    print("=" * 50)
    
    env.close()
    return agent, results


def evaluate_agent(weights_path, n_episodes=10, render=True):
    env = gym.make('LunarLander-v2', render_mode='human' if render else None)
    N_actions = env.action_space.n
    observation, info = env.reset(seed=42)
    N_state = len(observation)
    
    parameters = {'N_state': N_state, 'N_actions': N_actions}
    agent = DQN(parameters=parameters)
    agent.load_weights(weights_path)
    
    rewards = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, epsilon=0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            
            if render:
                env.render()
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    print(f"\nMean reward over {n_episodes} episodes: {sum(rewards) / len(rewards):.2f}")
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Lunar Lander agent")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train",
                        help="Train a new agent or evaluate existing one")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes")
    parser.add_argument("--weights", type=str, default="models/ddqn_weights.pth",
                        help="Path to weights file (for evaluation)")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save trained model")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering during evaluation")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(n_episodes=args.episodes, save_dir=args.save_dir)
    else:
        evaluate_agent(args.weights, render=not args.no_render)
