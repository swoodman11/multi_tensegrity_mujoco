# from tensegrity_env import TensegrityEnv
# import numpy as np

# # Test current implementation
# env = TensegrityEnv(obs_dim=104, visualize=False)
# print(f"Environment obs_dim: {env.sim.obs_dim}")
# print(f"Environment observation space: {env.observation_space.shape}")

# obs, _ = env.reset()
# print(f"Actual observation shape: {obs.shape}")
# print(f"Non-zero values: {np.count_nonzero(obs)}")

# # Check if get_observation method exists
# if hasattr(env.sim, 'get_observation'):
#     print("✅ get_observation method exists")
# else:
#     print("❌ get_observation method missing")

######################### second debugging script
# from tensegrity_env import TensegrityEnv
# import numpy as np

# env = TensegrityEnv(obs_dim=104, visualize=False)

# print("=== DEBUGGING OBSERVATION IMPLEMENTATION ===")

# # Test the observation method directly
# obs_direct = env.sim.get_observation()
# print(f"Direct get_observation() call: {obs_direct.shape}")

# # Check if helper methods exist
# helper_methods = ['get_robot_state', 'get_cable_state', 'get_end_effector_state', 
#                  'get_joint_velocities', 'get_contact_forces', 'get_time_features']

# for method_name in helper_methods:
#     if hasattr(env.sim, method_name):
#         try:
#             result = getattr(env.sim, method_name)()
#             print(f"✅ {method_name}(): {len(result)} dims")
#         except Exception as e:
#             print(f"❌ {method_name}(): Error - {e}")
#     else:
#         print(f"❌ {method_name}(): Not found")

# # Check what's actually in the observation
# obs, _ = env.reset()
# print(f"\nObservation analysis:")
# print(f"Shape: {obs.shape}")
# print(f"Min value: {obs.min():.3f}")
# print(f"Max value: {obs.max():.3f}")
# print(f"First 10 values: {obs[:10]}")

####################### third debugging script

from tensegrity_env import TensegrityEnv
import numpy as np

env = TensegrityEnv(obs_dim=104, visualize=True)

print("=== DEBUGGING ANTI-EXPLOITATION PENALTIES ===")

obs, _ = env.reset()

for step in range(20):
    # Use moderate action (not extreme)
    action = np.array([0.5] * 12)  # Middle range
    
    obs, reward, done, truncated, info = env.step(action)
    
    # Check if anti-exploit penalties exist and what they return
    if hasattr(env.sim, 'calculate_anti_exploit_penalties'):
        try:
            robot_pos = env.sim.get_robot_position()
            penalties = env.sim.calculate_anti_exploit_penalties(robot_pos, action)
            print(f"Step {step}: Reward={reward:.3f}, Anti-exploit penalty={penalties:.3f}")
            
            if abs(penalties) > 10:
                print(f"⚠️ LARGE PENALTY DETECTED: {penalties:.3f}")
        except Exception as e:
            print(f"Step {step}: Error calculating penalties - {e}")
    
    if abs(reward) > 100:
        print(f"⚠️ EXTREME REWARD: {reward:.3f}")
        break
    
    if done or truncated:
        print("Episode ended early")
        break

env.close()