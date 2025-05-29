"""
Analyzes correlation between communication messages and rewards or agent success.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils.config import config

def load_comm_logs(log_path):
    messages, actions, rewards, agents, steps = [], [], [], [], []
    with open(log_path, "r") as fin:
        for line in fin:
            entry = json.loads(line)
            messages.append(entry["message"])
            actions.append(entry["action"])
            rewards.append(entry["reward"])
            agents.append(entry["agent"])
            steps.append(entry["step"])
    return (
        np.array(messages),
        np.array(actions),
        np.array(rewards),
        np.array(agents),
        np.array(steps)
    )

def correlation_message_reward(messages, rewards, out_path):
    # L2 norm of message vector
    norms = np.linalg.norm(messages, axis=1)
    plt.figure(figsize=(6, 4))
    plt.scatter(norms, rewards, alpha=0.5)
    plt.xlabel("Message Norm")
    plt.ylabel("Reward")
    plt.title("Correlation: Message Norm vs. Reward")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    # Pearson correlation
    corr, pval = pearsonr(norms, rewards)
    print(f"Pearson r (norms vs. rewards): {corr:.3f} (p={pval:.4f})")

def main():
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    llm_log = os.path.join(config["logging"]["log_dir"], "comm_logs_llm.jsonl")
    if not os.path.exists(llm_log):
        print("LLM log not found.")
        return
    messages, actions, rewards, agents, steps = load_comm_logs(llm_log)
    correlation_message_reward(messages, rewards, os.path.join(results_dir, "corr_msgnorm_reward.png"))
    print("Saved correlation plot and stats to results/")

if __name__ == "__main__":
    main()
