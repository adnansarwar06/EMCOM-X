"""
MARL training/evaluation with LLM-driven communication for EMCOM-X.
"""

import ray
import numpy as np
from gymnasium.spaces import Box
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.mpe import simple_spread_v3
from models.commnet_model import CommNetModel
from utils.logging import MessageLogger
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import logging
from utils.config import config
from utils.llm_comm import load_llm, generate_message_from_obs, parse_llm_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Runs MARL with LLM comm integration."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("commnet_model", CommNetModel)
    env = simple_spread_v3.parallel_env()
    register_env("simple_spread", lambda config: simple_spread_v3.parallel_env())

    comm_size = config["commnet"]["comm_size"]
    obs_dim = config["marl"]["obs_dim"]
    combined_obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim + comm_size,),
        dtype=np.float32,
    )

    algo_config = (
        PPOConfig()
        .environment("simple_spread")
        .framework("torch")
        .training(
            model={
                "custom_model": "commnet_model",
                "custom_model_config": {"comm_size": comm_size},
            }
        )
        .multi_agent(
            policies={
                "default_policy": PolicySpec(
                    policy_class=None,
                    observation_space=combined_obs_space,
                    action_space=env.action_space(env.possible_agents[0]),
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "default_policy",
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = algo_config.build()
    env = simple_spread_v3.parallel_env()
    agent_ids = env.possible_agents

    logger_obj = MessageLogger(log_dir=config["logging"]["log_dir"])
    num_episodes = config["marl"]["num_episodes"]
    comm_size = config["commnet"]["comm_size"]

    # Load LLM just once
    tokenizer, model = load_llm()

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = {agent: False for agent in agent_ids}
        step = 0
        messages = {agent: None for agent in agent_ids}

        while not all(done.values()):
            actions = {}
            for agent in agent_ids:
                obs_vec = np.asarray(obs[agent]).flatten()
                # Generate comm vector using LLM
                try:
                    llm_output = generate_message_from_obs(
                        obs_vec,
                        tokenizer=tokenizer,
                        model=model
                    )
                    comm = parse_llm_message(llm_output, comm_size)
                except Exception as e:
                    logger.warning(f"LLM comm generation failed: {e}")
                    comm = np.zeros(comm_size, dtype=np.float32)

                full_obs = np.concatenate([obs_vec, comm]).astype(np.float32)
                policy = algo.get_policy("default_policy")
                action, _, extra = policy.compute_single_action(full_obs)
                actions[agent] = action
                messages[agent] = comm

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            for agent in agent_ids:
                logger_obj.log_step(
                    episode_id=episode,
                    step=step,
                    agent_id=agent,
                    obs=obs[agent],
                    message=messages[agent],
                    action=actions[agent],
                    reward=rewards[agent]
                )

            obs = next_obs
            done = {agent: terminated[agent] or truncated[agent] for agent in agent_ids}
            step += 1

        logger.info(f"Episode {episode + 1}/{num_episodes} finished.")

    logger_obj.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
