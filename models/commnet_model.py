"""
CommNetModel: Custom PyTorch model for multi-agent communication.

Implements observation + comm input to action, value, and message outputs for RLlib.
"""

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
import logging
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommNetModel(TorchModelV2, nn.Module):
    """
    Custom PyTorch model for MARL agent communication with RLlib.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, comm_size=None):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if comm_size is None:
            comm_size = config["commnet"]["comm_size"]
        self.input_size = obs_space.shape[0]
        self.comm_size = comm_size
        self.obs_size = self.input_size - self.comm_size

        logger.info(f"CommNetModel expects input_size={self.input_size} ({self.obs_size}+{self.comm_size})")

        self.fc1 = nn.Linear(self.input_size, config["commnet"]["hidden_dim"])
        self.fc2 = nn.Linear(config["commnet"]["hidden_dim"], config["commnet"]["hidden_dim"])
        self.policy_head = nn.Linear(config["commnet"]["hidden_dim"], num_outputs)
        self.value_head = nn.Linear(config["commnet"]["hidden_dim"], 1)
        self.message_head = nn.Linear(config["commnet"]["hidden_dim"], self.comm_size)

        self.last_message = None

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for RLlib.
        """
        x = input_dict["obs"].float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_size:
            if x.shape[1] > self.input_size:
                logger.warning(f"Input too long, truncating from {x.shape[1]} to {self.input_size}")
                x = x[:, :self.input_size]
            else:
                raise ValueError(f"Input shape too short: {x.shape}, expected {self.input_size}")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        message = self.message_head(x).detach().clone()
        self.last_message = message
        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(1)
        return logits, state

    def value_function(self):
        """
        Returns state-value estimation.
        """
        return self._value_out

    def get_last_message(self):
        """
        Returns the last message vector output by the model.
        """
        return self.last_message
