"""
LLM-based communication module for EMCOM-X.

Provides functions for generating and parsing communication messages from a fine-tuned LLM.
"""

import logging
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_llm():
    """
    Loads the tokenizer and model for the LLM from config.
    Returns:
        (tokenizer, model)
    """
    model_dir = config["llm"]["model_dir"]
    logger.info(f"Loading LLM from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model

def generate_message_from_obs(obs, tokenizer=None, model=None, context=None, max_new_tokens=None):
    """
    Generates a communication message from observation using the LLM.

    Args:
        obs (np.ndarray): Observation vector.
        tokenizer: Tokenizer instance.
        model: Model instance.
        context (str, optional): Extra context for the prompt.
        max_new_tokens (int, optional): Max tokens to generate.

    Returns:
        str: The raw output from the LLM.
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_llm()
    if max_new_tokens is None:
        max_new_tokens = config["llm"]["max_new_tokens"]
    obs_str = "[" + ", ".join(f"{x:.4f}" for x in obs) + "]"
    if context is None:
        prompt = f"OBS: {obs_str}"
    else:
        prompt = f"OBS: {obs_str} CONTEXT: {context}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"LLM Output: {result}")
    return result

def parse_llm_message(llm_output, comm_size):
    """
    Parses the LLM string output into a float numpy vector of the desired comm size.

    Args:
        llm_output (str): Raw string from the LLM.
        comm_size (int): Desired size of communication vector.

    Returns:
        np.ndarray: Communication vector (float, length comm_size).
    """
    msg_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", llm_output)
    msg_vector = [float(x) for x in msg_numbers]
    # Pad or crop to correct length
    if len(msg_vector) < comm_size:
        msg_vector += [0.0] * (comm_size - len(msg_vector))
    elif len(msg_vector) > comm_size:
        msg_vector = msg_vector[:comm_size]
    return np.array(msg_vector, dtype=np.float32)
