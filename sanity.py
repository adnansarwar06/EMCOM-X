"""
Sanity tests for EMCOM-X components: CommNetModel, MessageLogger, and PettingZoo environment integration.
"""

import torch
import gymnasium.spaces
import numpy as np
import logging
from models.commnet_model import CommNetModel
from utils.logging import MessageLogger
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pettingzoo.mpe import simple_spread_v3
except ImportError:
    logger.error("PettingZoo not installed. Please pip install pettingzoo")
    exit(1)

def test_commnet():
    """
    Tests basic CommNetModel instantiation and forward pass.
    """
    logger.info("Testing CommNetModel...")
    obs_size = config["commnet_test"]["obs_size"]
    comm_size = config["commnet_test"]["comm_size"]
    n_actions = config["commnet_test"]["n_actions"]

    obs_space = gymnasium.spaces.Box(low=0, high=1, shape=(obs_size,))
    action_space = gymnasium.spaces.Discrete(n_actions)
    model = CommNetModel(obs_space, action_space, num_outputs=n_actions, model_config={}, name="test", comm_size=comm_size)
    dummy_obs = torch.rand((1, obs_size))
    dummy_comm = torch.rand((1, comm_size))
    input_dict = {"obs": dummy_obs, "comm": dummy_comm}
    logits, state = model(input_dict, state=[], seq_lens=None)
    logger.info(f"Logits: {logits}")
    logger.info(f"Message vector: {model.get_last_message()}")
    assert logits.shape[-1] == n_actions, "Output logits shape mismatch"
    assert model.get_last_message() is not None, "Message vector missing"
    logger.info("CommNetModel test passed.")

def test_logger():
    """
    Tests MessageLogger functionality.
    """
    logger.info("Testing MessageLogger...")
    log_dir = config["logging"]["log_dir"]
    logger_obj = MessageLogger(log_dir=log_dir)
    logger_obj.log_step(episode_id=1, step=0, agent_id="agent_0", obs=[1,2,3], message=[0.5]*8, action=2, reward=1.0)
    logger_obj.close()
    with open(logger_obj.log_path, "r") as f:
        content = f.read()
        logger.info(f"Log file content: {content}")
    assert "agent_0" in content and "message" in content, "Logger output missing fields"
    logger.info("MessageLogger test passed.")

def test_env_with_commnet_and_logger():
    """
    Integration test: PettingZoo environment, CommNetModel, and MessageLogger.
    """
    logger.info("Testing PettingZoo env + CommNet + Logger integration...")
    comm_size = config["commnet_test"]["comm_size"]
    env = simple_spread_v3.parallel_env()
    obs, info = env.reset()
    agent_ids = env.possible_agents

    models = {}
    for agent in agent_ids:
        obs_array = np.array(obs[agent])
        logger.debug(f"Agent {agent} obs shape: {obs_array.shape}")
        obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=obs_array.shape)
        action_space = gymnasium.spaces.Discrete(env.action_space(agent).n)
        models[agent] = CommNetModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config={},
            name=f"commnet_{agent}",
            comm_size=comm_size
        )

    logger_obj = MessageLogger(log_dir=config["logging"]["log_dir"])
    messages = {agent: torch.zeros(comm_size) for agent in agent_ids}
    done = {agent: False for agent in agent_ids}
    step = 0

    while not all(done.values()) and step < config["sanity"]["max_steps"]:
        actions = {}
        for agent in agent_ids:
            obs_tensor = torch.tensor(obs[agent]).unsqueeze(0).float()
            comm_tensor = messages[agent].unsqueeze(0)
            input_dict = {"obs": obs_tensor, "comm": comm_tensor}
            logits, _ = models[agent](input_dict, state=[], seq_lens=None)
            action = int(torch.argmax(logits).item())
            actions[agent] = action
            msg_vec = models[agent].get_last_message()
            messages[agent] = msg_vec.squeeze(0).detach() if msg_vec is not None else torch.zeros(comm_size)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        for agent in agent_ids:
            logger_obj.log_step(
                episode_id=0,
                step=step,
                agent_id=agent,
                obs=obs[agent],
                message=messages[agent].tolist(),
                action=actions[agent],
                reward=rewards[agent]
            )
        obs = next_obs
        done = {agent: terminated[agent] or truncated[agent] for agent in agent_ids}
        step += 1
    logger_obj.close()
    logger.info("Integration test finished. Check logs directory for output.")
    logger.info("Env + CommNet + Logger integration test passed.")

if __name__ == "__main__":
    test_commnet()
    test_logger()
    test_env_with_commnet_and_logger()
    logger.info("All sanity tests completed successfully.")
