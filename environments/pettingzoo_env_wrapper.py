"""
Environment wrapper utilities for PettingZoo MPE environments.

Defines make_env for use with RLlib.
"""

from pettingzoo.mpe import simple_spread_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from utils.config import config

def make_env(env_name=None, env_config=None):
    """
    Returns a wrapped PettingZoo environment for use with RLlib.

    Args:
        env_name (str): Name of the environment. Uses config if not provided.
        env_config (dict): Environment-specific config (optional).

    Returns:
        ParallelPettingZooEnv: RLlib-compatible parallel PettingZoo env.
    """
    if env_name is None:
        env_name = config["env"]["env_name"]

    env_config = env_config or config["env"].get("env_config", {})

    if env_name == "simple_spread_v3":
        env = simple_spread_v3.parallel_env(**env_config)
    else:
        raise NotImplementedError(f"Unknown environment: {env_name}")

    return ParallelPettingZooEnv(env)
