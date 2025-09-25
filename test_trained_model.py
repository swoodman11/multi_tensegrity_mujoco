# from stable_baselines3 import PPO
# from tensegrity_env import TensegrityEnv
# import time

# # Load the trained model
# model = PPO.load("ppo_tensegrity_gait")
# env = TensegrityEnv()

# # Test for 3 episodes
# for episode in range(3):
#     obs, info = env.reset()
#     total_reward = 0
#     steps = 0
    
#     print(f"\n=== Episode {episode + 1} ===")
    
#     for step in range(1000):  # Max steps per episode  #tensorboard --logdir=./ppo_tensegrity_tensorboard/ --port=6007
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
        
#         total_reward += reward
#         steps += 1
        
#         # Render if you want to see it (might be slow)
#         # env.render()
#         # time.sleep(0.01)
        
#         if done or truncated:
#             break
    
#     print(f"Episode {episode + 1}: {steps} steps, Total reward: {total_reward:.2f}")

# # env.close()

from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
import time

# Load the trained model
model = PPO.load("ppo_tensegrity_gait_20250925_155047")
env = TensegrityEnv(visualize=True)  # Now created with visualization enabled

print("Testing trained model with visualization...")
print("Press 'q' in the render window to quit early")

# Test for 1 episode with rendering
obs, info = env.reset()
total_reward = 0
steps = 0

print(f"\n=== Visualizing Robot Gait ===")

for step in range(1000):  # Max steps per episode
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    # Render the robot
    env.render()
    time.sleep(0.05)  # Slow down for better viewing
    
    # Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.2f}")
    
    if done or truncated:
        print("Episode ended!")
        break

print(f"Final: {steps} steps, Total reward: {total_reward:.2f}")

# Keep window open for a bit
time.sleep(2)

# Close properly
try:
    import cv2
    cv2.destroyAllWindows()
except:
    pass

#Zac: run this for tensor flow charts: tensorboard --logdir=./ppo_tensegrity_tensorboard/ --port=6007