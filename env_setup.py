"""
Project environment setup utility for EMCOM-X.

Creates necessary directory structure for the project, including logs and results folders.
Writes a README.md in each directory for documentation purposes.
"""

import os
import logging
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_project_dirs(dirs):
    """
    Ensures required directories exist and each contains a README.md file.

    Args:
        dirs (list): List of directory names (str) to create.
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")
        else:
            logger.info(f"Directory already exists: {d}")

        readme_path = os.path.join(d, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# {d} directory\n\nAuto-created by env_setup.py.\n")
            logger.info(f"Created {readme_path}")
        else:
            logger.info(f"README.md already exists in {d}")

def main():
    """
    Main entry point for environment setup.
    """
    dirs = config.get("project_dirs", [])
    create_project_dirs(dirs)
    logger.info("Project structure ready for use.")

if __name__ == "__main__":
    main()
