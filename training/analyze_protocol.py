"""
Protocol emergence: clusters LLM messages and maps them to actions/rewards/outcomes.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
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

def cluster_messages(messages, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(messages)
    return cluster_labels, kmeans

def plot_clusters(messages, cluster_labels, actions, rewards, out_dir):
    # Reduce to 2D for plotting
    pca = PCA(n_components=2)
    proj = pca.fit_transform(messages)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.title("LLM Message Clusters (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_clusters_pca.png"))
    plt.close()

    # Cluster-to-action mapping
    plt.figure(figsize=(7, 4))
    for cl in np.unique(cluster_labels):
        cl_actions = actions[cluster_labels == cl]
        plt.hist(cl_actions, bins=np.arange(actions.max()+2)-0.5, alpha=0.5, label=f"Cluster {cl}")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Distribution by Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_action_distribution.png"))
    plt.close()

    # Cluster-to-reward mapping
    reward_means = []
    for cl in np.unique(cluster_labels):
        cl_rewards = rewards[cluster_labels == cl]
        reward_means.append(np.mean(cl_rewards))
    plt.figure(figsize=(6, 4))
    plt.bar(np.unique(cluster_labels), reward_means)
    plt.xlabel("Cluster")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward by Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_mean_reward.png"))
    plt.close()

def main():
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    llm_log = os.path.join(config["logging"]["log_dir"], "comm_logs_llm.jsonl")
    if not os.path.exists(llm_log):
        print("LLM log not found.")
        return
    messages, actions, rewards, agents, steps = load_comm_logs(llm_log)
    cluster_labels, _ = cluster_messages(messages, n_clusters=8)
    plot_clusters(messages, cluster_labels, actions, rewards, results_dir)
    print(f"Saved cluster plots and analysis to {results_dir}/")

if __name__ == "__main__":
    main()
