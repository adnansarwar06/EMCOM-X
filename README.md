# EMCOM-X: Multi-Agent Communication via RLlib and LLMs

EMCOM-X is a modular research platform for exploring communication in multi-agent reinforcement learning (MARL). The system allows agents to communicate either with traditional neural message vectors or with messages generated by a fine-tuned large language model (LLM). The platform includes tools for training, detailed logging, and post-hoc analysis of agent communication patterns.

## How It Works

- Agents interact in a PettingZoo environment (default: `simple_spread_v3`).
- At each step, agents communicate using either dense neural vectors (baseline) or LLM-generated message vectors.
- All agent observations, messages, actions, and rewards are logged for later analysis.
- Analysis scripts are provided to visualize communication patterns and compare the effect of LLM communication versus the baseline.

## How to Run

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Edit `utils/config.py`** to set experiment parameters and specify the path to your fine-tuned LLM if using LLM-based comms.

3. **To run a baseline experiment (no LLM):**
    ```bash
    python -m training.marl_train_baseline
    ```

4. **To run with LLM-based communication:**
    ```bash
    python -m training.marl_train_llm
    ```

5. **To analyze results and generate plots:**
    ```bash
    python -m training.analyze_comm
    python -m training.analyze_protocol
    python -m training.plot_trajectories
    python -m training.correlation_analysis
    ```

Logs will be written to the `logs/` directory, and analysis figures to `results/`.

---
