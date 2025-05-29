"""
Analyzes communication logs for EMCOM-X: clustering, dimensionality reduction, and reward comparison.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.config import config

def load_comm_logs(log_path):
    """
    Loads all messages, actions, and rewards from a JSONL log file.
    Returns:
        messages (np.ndarray), actions (np.ndarray), rewards (np.ndarray)
    """
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

def plot_comm_vectors(messages, title, out_path):
    """
    Plots 2D PCA and t-SNE of the comm vectors.
    """
    # PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(messages)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_proj[:, 0], pca_proj[:, 1], alpha=0.5)
    plt.title(f"{title} (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{out_path}_pca.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    tsne_proj = tsne.fit_transform(messages)
    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], alpha=0.5)
    plt.title(f"{title} (t-SNE)")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.tight_layout()
    plt.savefig(f"{out_path}_tsne.png")
    plt.close()

def plot_rewards(rewards, title, out_path):
    """
    Plots rewards per step or episode.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, alpha=0.7)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    """
    Loads, analyzes, and visualizes both baseline and LLM logs.
    """
    log_dir = config["logging"]["log_dir"]
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # You may need to set these filenames after a run:
    baseline_log = os.path.join(log_dir, "comm_logs_baseline.jsonl")
    llm_log = os.path.join(log_dir, "comm_logs_llm.jsonl")

    for log_path, label in [(baseline_log, "Baseline"), (llm_log, "LLM")]:
        if not os.path.exists(log_path):
            print(f"Log {log_path} not found. Skipping.")
            continue
        messages, actions, rewards, agents, steps = load_comm_logs(log_path)
        plot_comm_vectors(messages, f"{label} Communication", os.path.join(results_dir, f"{label.lower()}_comm"))
        plot_rewards(rewards, f"{label} Reward per Step", os.path.join(results_dir, f"{label.lower()}_rewards.png"))

        # Print summary stats
        print(f"\n--- {label} Stats ---")
        print(f"Mean Reward: {np.mean(rewards):.3f}")
        print(f"Std Reward: {np.std(rewards):.3f}")
        print(f"Mean Message Norm: {np.mean(np.linalg.norm(messages, axis=1)):.3f}")

if __name__ == "__main__":
    main()
