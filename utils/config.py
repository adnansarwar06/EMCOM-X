config = {
    "env": {
        "env_name": "simple_spread_v3",
        "env_config": {},
    },
    "project_dirs": [
        "environments",
        "models",
        "training",
        "agents",
        "retrieval",
        "utils",
        "notebooks",
        "logs",
        "results"
    ],
    "logging": {
        "log_dir": "logs"
    },
    "commnet_test": {
        "obs_size": 4,
        "comm_size": 8,
        "n_actions": 5
    },
    "sanity": {
        "max_steps": 3
    },
    "llm": {
        "model_dir": "models/llm_lora_model",
        "max_new_tokens": 32,
        "obs_example": "[0.0, -0.5, 0.3, ...]"
    },
    "llm_data": {
        "log_file_path": "logs/comm_logs_latest.jsonl",
        "output_file_path": "llm_train.txt"
    },
    "commnet": {
        "comm_size": 8,
        "hidden_dim": 64
    },
    "marl": {
        "obs_dim": 18,           # Change as appropriate for your environment
        "num_episodes": 500
    },
    "ppo": {
        "train_batch_size": 4000,
        "num_gpus": 0,
        "num_iters": 3
    },
    "results_dir": "results",
    "seed": 42,
    "baseline_log": "comm_logs_baseline.jsonl",
    "llm_log": "comm_logs_llm.jsonl",
    "experiment_name": "emcomx_run1"
}
