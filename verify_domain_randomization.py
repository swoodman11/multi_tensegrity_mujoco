"""
Verification script to check domain randomization implementation.
Run this to verify all changes are correctly applied.
"""

import numpy as np
from tensegrity_env import TensegrityEnv
from pathlib import Path

print("="*70)
print("DOMAIN RANDOMIZATION VERIFICATION")
print("="*70)

# 1. Check observation space dimensions
print("\n1. Checking observation space...")
env = TensegrityEnv(visualize=False)
obs_shape = env.observation_space.shape[0]
expected_dim = 98 if env.sim.obs_mode == "tier2" else 106

print(f"   Observation mode: {env.sim.obs_mode}")
print(f"   Observation space dimension: {obs_shape}")
print(f"   Expected dimension: {expected_dim}")
if obs_shape == expected_dim:
    print("   ✓ PASS: Observation space dimension is correct")
else:
    print(f"   ✗ FAIL: Expected {expected_dim}, got {obs_shape}")

# 2. Check observation includes goal direction
print("\n2. Checking observation includes goal direction...")
obs, info = env.reset()
obs_actual_shape = obs.shape[0]
print(f"   Actual observation shape: {obs_actual_shape}")
print(f"   Goal direction: {env.sim.goal_direction}")
if obs_actual_shape == expected_dim:
    print("   ✓ PASS: Observation shape matches expected dimension")
else:
    print(f"   ✗ FAIL: Observation shape is {obs_actual_shape}, expected {expected_dim}")

# 3. Check goal direction is in observation
print("\n3. Verifying goal direction in observation...")
goal_dir_in_obs = obs[-2:]  # Last 2 elements should be goal direction
goal_dir_actual = env.sim.goal_direction
diff = np.linalg.norm(goal_dir_in_obs - goal_dir_actual)
print(f"   Last 2 obs elements: {goal_dir_in_obs}")
print(f"   Actual goal direction: {goal_dir_actual}")
print(f"   Difference (L2 norm): {diff:.6f}")
if diff < 0.01:
    print("   ✓ PASS: Goal direction correctly appended to observation")
else:
    print(f"   ✗ FAIL: Goal direction mismatch (diff={diff})")

# 4. Check domain randomization on reset
print("\n4. Testing domain randomization on multiple resets...")
yaw_angles = []
friction_multipliers = []
goal_directions = []

for i in range(5):
    obs, info = env.reset()
    yaw = env.sim.initial_yaw
    goal_dir = env.sim.goal_direction.copy()
    
    yaw_angles.append(yaw)
    goal_directions.append(goal_dir)
    
    print(f"   Reset {i+1}: yaw={yaw:.3f} rad ({np.degrees(yaw):.1f}°), goal_dir=[{goal_dir[0]:.3f}, {goal_dir[1]:.3f}]")

yaw_std = np.std(yaw_angles)
print(f"\n   Yaw angle std dev: {yaw_std:.3f} rad ({np.degrees(yaw_std):.1f}°)")
if yaw_std > 0.5:  # Should have significant variation
    print("   ✓ PASS: Yaw angles are randomized")
else:
    print("   ✗ FAIL: Yaw angles show insufficient randomization")

# 5. Check goal direction normalization
print("\n5. Checking goal direction is normalized...")
all_normalized = True
for i, goal_dir in enumerate(goal_directions):
    norm = np.linalg.norm(goal_dir)
    if abs(norm - 1.0) > 0.01:
        print(f"   Reset {i+1}: ✗ FAIL - norm={norm:.6f} (not unit vector)")
        all_normalized = False
    else:
        print(f"   Reset {i+1}: ✓ norm={norm:.6f}")

if all_normalized:
    print("   ✓ PASS: All goal directions are normalized")
else:
    print("   ✗ FAIL: Some goal directions are not unit vectors")

# 6. Check simulator attributes
print("\n6. Checking simulator has required attributes...")
required_attrs = ['goal_direction', 'initial_yaw', 'cumulative_distance']
all_present = True
for attr in required_attrs:
    if hasattr(env.sim, attr):
        value = getattr(env.sim, attr)
        print(f"   ✓ {attr}: {value}")
    else:
        print(f"   ✗ {attr}: NOT FOUND")
        all_present = False

if all_present:
    print("   ✓ PASS: All required attributes present")
else:
    print("   ✗ FAIL: Some attributes missing")

# 7. Test a short episode
print("\n7. Running short test episode...")
obs, info = env.reset()
initial_goal = env.sim.goal_direction.copy()
total_reward = 0
print(f"   Initial goal direction: [{initial_goal[0]:.3f}, {initial_goal[1]:.3f}]")

for step in range(10):
    action = np.random.uniform(0, 1, size=12)  # Random action
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"   Episode completed: 10 steps, total reward={total_reward:.3f}")
print(f"   Final goal direction: [{env.sim.goal_direction[0]:.3f}, {env.sim.goal_direction[1]:.3f}]")

# Check goal direction didn't change during episode
goal_changed = np.linalg.norm(env.sim.goal_direction - initial_goal)
if goal_changed < 0.01:
    print("   ✓ PASS: Goal direction remains constant during episode")
else:
    print(f"   ✗ FAIL: Goal direction changed during episode (diff={goal_changed})")

# 8. Check info dict has directional reward components
print("\n8. Checking info dict for directional reward components...")
required_keys = ['directional_velocity_reward', 'perpendicular_penalty', 'cumulative_distance_bonus']
all_keys_present = True
for key in required_keys:
    if key in info:
        print(f"   ✓ {key}: {info[key]:.4f}")
    else:
        print(f"   ✗ {key}: NOT FOUND")
        all_keys_present = False

if all_keys_present:
    print("   ✓ PASS: All directional reward components in info dict")
else:
    print("   ✗ FAIL: Some reward components missing")

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print("Run this script to verify domain randomization is correctly implemented.")
print("All checks should show ✓ PASS.")
print("\nIf all checks pass, you can proceed with training:")
print("  python gpu_pretraining_SAC.py")
print("\nTo test directional control after training:")
print("  python test_trained_model_SAC.py --test-directions")
print("="*70)
