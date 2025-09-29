from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tensegrity_env import TensegrityEnv
import os
from datetime import datetime

def train_parallel():
    # Configuration
    n_envs = 4  # Start with fewer environments to debug
    total_timesteps = 100_000
    
    print(f"Training with {n_envs} parallel environments...")
    
    # Create environment factory functions
    def make_env_factory(rank):
        def make_env():
            # Only rank 0 gets visualization
            visualize = (rank == 0)
            return TensegrityEnv(obs_dim=78, visualize=visualize)
        return make_env
    
    # Create list of environment factories
    env_factories = [make_env_factory(i) for i in range(n_envs)]
    
    # Use DummyVecEnv first to test (single process, easier debugging)
    env = DummyVecEnv(env_factories)
    
    # If DummyVecEnv works, then try SubprocVecEnv for true parallelism:
    # env = SubprocVecEnv(env_factories)
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        tensorboard_log="./ppo_tensegrity_tensorboard/",
        device="cpu"  # Start with CPU to avoid CUDA issues
    )
    
    print("Model created successfully!")
    
    # Train the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"parallel_run_{timestamp}"
    )
    
    # Save the trained model
    model.save(f"ppo_tensegrity_parallel_{timestamp}")
    print(f"Model saved as ppo_tensegrity_parallel_{timestamp}")
    
    env.close()

if __name__ == "__main__":
    train_parallel()