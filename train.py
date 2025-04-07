'''CARLA autonomous driving reinforcement learning training script.

This module implements a complete training process for training CARLA autonomous driving agents. Main functions include:

- Training using the PPO (Proximal Policy Optimization) algorithm
- Support for Tensorboard real-time monitoring of the training process
- Provide model checkpoint saving and optimal model saving
- Regular evaluation to track training progress
- Support video recording to visualize agent behavior

Typical usage:
    python train.py  # Start training, using default configuration
'''

import argparse
from datetime import datetime
import os
import time

import carla
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import CarlaEnv

def make_env():
    '''Creates and configures the CARLA environment for training.

    Creates a basic CARLA environment instance and wraps it with a Monitor wrapper to record training data.
    Ensures that the log directory exists, and all monitoring data will be saved in that directory.

    Returns:
        Monitor: The wrapped CARLA environment, ready for training and evaluation.
    '''
    # Create the basic CARLA environment
    env = CarlaEnv()
    
    # Set the monitoring log directory
    log_dir = os.path.join("logs", "monitor")
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrap the environment with Monitor and return
    return Monitor(env, log_dir)

def main():
    '''Executes the complete CARLA autonomous driving agent training process.

    Main steps:
        1. Set up log directories and Tensorboard configuration
        2. Create and configure the training environment
        3. Initialize the PPO model and set training parameters
        4. Configure model checkpoints and evaluation callbacks
        5. Execute the training loop
        6. Save the final trained model
    
    Exception handling:
        - Supports graceful interruption of training via Ctrl+C
        - Ensures proper cleanup of resources in case of exceptions
    '''
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="CARLA autonomous driving reinforcement learning training script")
    
    # Add checkpoint_path parameter
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
    
    # Parse command line arguments
    args = parser.parse_args()

    # Set the training log directory
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"PPO_{current_time}")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the training environment
    env = DummyVecEnv([lambda: CarlaEnv()])  # Vectorized environment wrapper

    # Check if a checkpoint file exists
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        # Load the checkpoint file
        model = PPO.load(args.checkpoint_path, env=env)
        print(f"Loaded checkpoint file: {args.checkpoint_path}")
    else:
        # Configure the PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,          # Learning rate
            n_steps=2048,               # Number of steps to collect for each update
            batch_size=64,              # Batch size for each optimization
            n_epochs=10,                # Number of training rounds for each update
            gamma=0.99,                 # Discount factor
            gae_lambda=0.95,            # GAE lambda parameter
            clip_range=0.2,             # PPO clip parameter
            ent_coef=0.01,              # Entropy coefficient, used to encourage exploration
            verbose=1,
            tensorboard_log=log_dir
        )
            
    # Set up callback functions
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="carla_model"
    )
    
    # Create an evaluation environment
    eval_env = DummyVecEnv([lambda: CarlaEnv()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=10000,             # Evaluate every 10000 steps
        n_eval_episodes=5,           # Evaluate 5 episodes each time
        deterministic=True
    )
    
    # Train the model
    total_timesteps = 10_000_000     # Total number of training steps
    try:
        print(f"Starting training, total steps: {total_timesteps}")
        print(f"You can view Tensorboard using the following command:")
        print(f"tensorboard --logdir={log_dir}")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save the final model
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"Training complete! Model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        # Save the interrupted model
        interrupted_model_path = os.path.join(log_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"Saved the interrupted model to: {interrupted_model_path}")
    
    finally:
        # Clean up the environment
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()