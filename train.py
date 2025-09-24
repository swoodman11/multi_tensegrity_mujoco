from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv

env = TensegrityEnv()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensegrity_tensorboard/",device='cpu')
model.learn(total_timesteps=2_000_000)

model.save("ppo_tensegrity_gait")
