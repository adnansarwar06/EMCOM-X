"""
Simple LLM inference test for EMCOM-X.

Loads a fine-tuned language model and generates a message from a given observation.
"""

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_message(obs, context=None, max_new_tokens=None):
    """
    Generates a message from the LLM based on the given observation and optional context.

    Args:
        obs (str): Observation string.
        context (str, optional): Extra context to provide to LLM.
        max_new_tokens (int, optional): Maximum number of new tokens for generation.

    Returns:
        str: The generated message.
    """
    if max_new_tokens is None:
        max_new_tokens = config["llm"]["max_new_tokens"]

    model_dir = config["llm"]["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if context is None:
        prompt = f"OBS: {obs}"
    else:
        prompt = f"OBS: {obs} CONTEXT: {context}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Full output: {result}")
    return result

if __name__ == "__main__":
    obs_example = config["llm"]["obs_example"]
    msg = generate_message(obs_example)
    logger.info(f"Generated message: {msg}")
