"""
Converts MARL communication log files to a supervised LLM fine-tuning dataset.

Each output line: <obs as str> \t <message as str>
"""

import json
import logging
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm_dataset_from_logs(log_file_path=None, output_file_path=None):
    """
    Processes a log file to generate LLM training data.

    Args:
        log_file_path (str, optional): Path to the JSONL log file. Uses config if None.
        output_file_path (str, optional): Output file for LLM dataset. Uses config if None.
    """
    if log_file_path is None:
        log_file_path = config["llm_data"]["log_file_path"]
    if output_file_path is None:
        output_file_path = config["llm_data"]["output_file_path"]

    count = 0
    with open(log_file_path, "r") as fin, open(output_file_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            obs = entry.get("obs")
            message = entry.get("message")
            # Flatten message if it's a list of list
            if isinstance(message, list) and len(message) == 1 and isinstance(message[0], list):
                message = message[0]
            inp = f"OBS: {obs}"
            tgt = str(message)
            fout.write(f"{inp}\t{tgt}\n")
            count += 1

    logger.info(f"Processed {count} examples to {output_file_path}")

if __name__ == "__main__":
    create_llm_dataset_from_logs()
