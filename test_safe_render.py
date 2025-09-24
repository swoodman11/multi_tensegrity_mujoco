from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
import numpy as np

model = PPO.load("ppo_tensegrity_gait")
env = TensegrityEnv(visualize=True)

obs, info = env.reset()
positions = []
rewards = []

print("Testing with safe rendering...")

try:
    for step in range(50):  # Shorter test to avoid segfault
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)
        
        # Get position data
        try:
            end_pts = env.sim.get_endpts()
            robot_pos = end_pts.mean(axis=0)
            positions.append([robot_pos[0], robot_pos[1], robot_pos[2]])
            
            if step % 10 == 0:
                print(f"Step {step}: Pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}), Reward={reward:.3f}")
        except Exception as e:
            print(f"Error getting position: {e}")
        
        # Try rendering (with error handling)
        try:
            frame = env.render(mode='rgb_array')  # Get frame without displaying
            if frame is not None and step % 10 == 0:
                print(f"Got frame shape: {frame.shape}")
        except Exception as e:
            print(f"Render error: {e}")
        
        if done or truncated:
            break
            
except Exception as e:
    print(f"Test failed: {e}")

# Show results
positions = np.array(positions)
if len(positions) > 0:
    print(f"\nResults:")
    print(f"Distance traveled: {positions[-1][0] - positions[0][0]:.3f} m")
    print(f"Average reward: {np.mean(rewards):.3f}")

env.close()