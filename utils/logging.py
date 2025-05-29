"""
Logging utility for EMCOM-X.

Implements MessageLogger class for per-step agent-environment interaction logging.
"""

import os
import json
import logging
from datetime import datetime
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageLogger:
    """
    MessageLogger writes step-wise logs (JSONL) for agent observations, messages, actions, and rewards.
    """
    def __init__(self, log_dir=None):
        """
        Initializes the logger, creating a unique log file.
        
        Args:
            log_dir (str, optional): Directory to save logs. Uses config if None.
        """
        if log_dir is None:
            log_dir = config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"comm_logs_{now}.jsonl")
        self.log_file = open(self.log_path, "a")
        logger.info(f"MessageLogger: Logging to {self.log_path}")

    def log_step(self, episode_id, step, agent_id, obs, message, action, reward):
        """
        Logs a single agent step as a JSONL entry.

        Args:
            episode_id (int): Episode number.
            step (int): Step number.
            agent_id (str): Agent identifier.
            obs (array-like): Agent's observation.
            message (array-like): Message vector.
            action (int): Chosen action.
            reward (float): Obtained reward.
        """
        entry = {
            "episode": int(episode_id),
            "step": int(step),
            "agent": str(agent_id),
            "obs": obs.tolist() if hasattr(obs, "tolist") else list(obs),
            "message": message.tolist() if hasattr(message, "tolist") else list(message),
            "action": int(action),
            "reward": float(reward),
        }
        self.log_file.write(json.dumps(entry) + "\n")

    def close(self):
        """Closes the log file."""
        self.log_file.close()
        logger.info(f"MessageLogger: Closed log at {self.log_path}")
