"""
Configuration parameters for the smart grid environment and training.
"""

# Environment Configuration
ENV_CONFIG = {
    "num_agents": 5,
    "max_steps": 288,  # 24 hours at 5-min intervals
    "max_energy_capacity": 100.0,  # kWh
    "max_transfer_rate": 20.0,  # kWh per timestep
    "base_energy_price": 0.15,  # $/kWh
    "grid_import_penalty": 1.5,
}

# Training Configuration
TRAINING_CONFIG = {
    "num_iterations": 500,
    "checkpoint_frequency": 50,
    "use_gpu": True,
    "num_rollout_workers": 4,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
}

# Model Configuration
MODEL_CONFIG = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
    "use_lstm": False,
}







