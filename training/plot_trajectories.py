"""
Plots agent actions, comm messages, and rewards over a single episode for sequence analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from utils.config import config
import os

def load_episode(log_path, episode=0):
    # Returns data for one episode (all steps, all agents)
    episode_data = []
    with open(log_path, "r") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry["episode"] == episode:
                episode_data.append(entry)
    return episode_data

def plot_trajectory(episode_data, out_dir, episode=0):
    agents = sorted(list(set(entry["agent"] for entry in episode_data)))
    steps = sorted(list(set(entry["step"] for entry in episode_data)))
    n_agents = len(agents)
    n_steps = len(steps)

    rewards = np.zeros((n_agents, n_steps))
    actions = np.zeros((n_agents, n_steps))
    messages = np.zeros((n_agents, n_steps, len(episode_data[0]["message"])))

    for entry in episode_data:
        a_idx = agents.index(entry["agent"])
        s_idx = entry["step"]
        rewards[a_idx, s_idx] = entry["reward"]
        actions[a_idx, s_idx] = entry["action"]
        messages[a_idx, s_idx] = np.array(entry["message"])

    # Plot actions
    plt.figure(figsize=(8, 4))
    for a_idx, agent in enumerate(agents):
        plt.plot(actions[a_idx], label=f"{agent}")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.title(f"Agent Actions (Episode {episode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"actions_episode_{episode}.png"))
    plt.close()

    # Plot rewards
    plt.figure(figsize=(8, 4))
    for a_idx, agent in enumerate(agents):
        plt.plot(rewards[a_idx], label=f"{agent}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Agent Rewards (Episode {episode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rewards_episode_{episode}.png"))
    plt.close()

    # Plot message norm
    plt.figure(figsize=(8, 4))
    for a_idx, agent in enumerate(agents):
        norms = np.linalg.norm(messages[a_idx], axis=1)
        plt.plot(norms, label=f"{agent}")
    plt.xlabel("Step")
    plt.ylabel("Message L2 Norm")
    plt.title(f"Agent Message Norms (Episode {episode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"messages_norm_episode_{episode}.png"))
    plt.close()

def main():
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    llm_log = os.path.join(config["logging"]["log_dir"], "comm_logs_llm.jsonl")
    episode = 0  # Change to plot other episodes
    episode_data = load_episode(llm_log, episode)
    if not episode_data:
        print(f"No data found for episode {episode}")
        return
    plot_trajectory(episode_data, results_dir, episode)
    print(f"Plotted episode {episode} trajectory in {results_dir}/")

if __name__ == "__main__":
    main()
