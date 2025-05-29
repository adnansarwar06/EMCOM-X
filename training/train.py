"""
Quick RLlib PPO training loop for EMCOM-X using a PettingZoo MPE environment.
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from environments.pettingzoo_env_wrapper import make_env
import logging
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Runs a short PPO training loop and saves a checkpoint."""
    logger.info("Starting Ray...")
    ray.init(ignore_reinit_error=True, include_dashboard=False, local_mode=True)

    env_name = config["env"]["env_name"]
    register_env(env_name, lambda config: make_env(env_name, config))

    train_batch_size = config["ppo"]["train_batch_size"]
    num_gpus = config["ppo"]["num_gpus"]
    num_iters = config["ppo"]["num_iters"]

    algo_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .env_runners(num_env_runners=0)   # Set 1+ for parallel speed in production
        .resources(num_gpus=num_gpus)
        .training(train_batch_size=train_batch_size)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
        )
    )

    logger.info("Building PPO algorithm...")
    algo = algo_config.build()

    logger.info("Running training loop...")
    for i in range(num_iters):
        result = algo.train()
        mean_return = result["env_runners"]["episode_return_mean"]
        agent_means = result["env_runners"]["agent_episode_returns_mean"]
        logger.info(f"Iter {i}: Episode Return Mean: {mean_return:.2f} | Agent Returns: {agent_means}")

    checkpoint = algo.save()
    logger.info(f"Checkpoint saved at {checkpoint}")

    ray.shutdown()

if __name__ == "__main__":
    main()
