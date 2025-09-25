from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
from datetime import datetime

# Set observation dimension
obs_dim = 36
env = TensegrityEnv(obs_dim=obs_dim)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensegrity_tensorboard/",device='cpu')
model.learn(total_timesteps=300_000)

# Generate timestamp for unique model filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"ppo_tensegrity_gait_{timestamp}"

model.save(model_filename)

print(f"Model saved as: {model_filename}.zip")
