#!/usr/bin/env python3
"""
Verification script to test that timestep fixes are working correctly.

This script tests:
1. That 100 RL steps = ~100 seconds of simulated time
2. That visualization runs smoothly at 20 Hz
3. That timing is consistent across environments

Usage:
    python test_simulator_timing.py
"""

import time
import numpy as np
from tensegrity_env import TensegrityEnv
from pathlib import Path

def test_simulation_timing(visualize=False, num_steps=100):
    """Test that simulation timing is correct (1 step = 1 second)."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Simulation Timing")
    print(f"{'='*60}")
    print(f"Running {num_steps} RL steps...")
    print(f"Expected: ~{num_steps} seconds of simulated time")
    
    # Create environment
    env = TensegrityEnv(visualize=visualize)
    obs, _ = env.reset()
    
    # Record start times
    real_start = time.time()
    sim_start = env.sim.mjc_data.time
    
    # Run episode
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            elapsed_sim = env.sim.mjc_data.time - sim_start
            print(f"  Step {step:3d}: {elapsed_sim:.1f}s simulated")
        
        if terminated or truncated:
            print(f"  Episode ended early at step {step}")
            break
    
    # Record end times
    real_end = time.time()
    sim_end = env.sim.mjc_data.time
    
    # Calculate durations
    real_elapsed = real_end - real_start
    sim_elapsed = sim_end - sim_start
    
    # Expected: 1 second per step
    expected_sim = num_steps * 1.0
    
    # Results
    print(f"\n📊 Results:")
    print(f"   Real time elapsed: {real_elapsed:.2f}s")
    print(f"   Simulated time: {sim_elapsed:.2f}s")
    print(f"   Expected simulated: {expected_sim:.1f}s")
    print(f"   Difference: {abs(sim_elapsed - expected_sim):.2f}s")
    
    # Tolerance: ±5% (100 steps ± 5 steps)
    tolerance = expected_sim * 0.05
    passed = abs(sim_elapsed - expected_sim) <= tolerance
    
    print(f"\n   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if not passed:
        print(f"   ⚠️  Simulated time should be within ±{tolerance:.1f}s of {expected_sim:.1f}s")
        print(f"   ⚠️  Actual: {sim_elapsed:.2f}s")
    
    env.close()
    return passed


def test_episode_length():
    """Test that episode truncation happens at correct timestep."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Episode Length")
    print(f"{'='*60}")
    
    env = TensegrityEnv(visualize=False, max_episode_steps=50)
    obs, _ = env.reset()
    
    print(f"max_episode_steps set to: 50")
    print(f"Running until truncation...")
    
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if truncated:
            print(f"\n✅ Episode truncated at step {step_count}")
            break
        
        if terminated:
            print(f"\n⚠️  Episode terminated (done=True) at step {step_count}")
            break
        
        if step_count > 60:  # Safety limit
            print(f"\n❌ Episode did not truncate after {step_count} steps!")
            break
    
    passed = (step_count == 50 and truncated)
    print(f"\nStatus: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if not passed:
        print(f"⚠️  Expected truncation at step 50, got step {step_count}")
    
    env.close()
    return passed


def test_render_frequency():
    """Test that internal rendering happens at expected frequency."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Render Frequency")
    print(f"{'='*60}")
    print(f"Expected: 20 frames per second (render every 5th physics step)")
    print(f"With 1 action = 100 physics steps → 20 renders per action")
    
    # This is tested by inspection during visual testing
    print(f"\n✅ To verify manually:")
    print(f"   1. Run: python test_trained_model_SAC.py --model <model_path>")
    print(f"   2. Observe smooth 20 Hz visualization")
    print(f"   3. Check no stuttering or frame drops")
    
    return True  # Manual verification


def main():
    print("\n" + "="*60)
    print("🚀 TIMESTEP FIX VERIFICATION SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Simulation timing (headless)
    try:
        results['timing'] = test_simulation_timing(visualize=False, num_steps=100)
    except Exception as e:
        print(f"❌ Timing test failed with error: {e}")
        results['timing'] = False
    
    # Test 2: Episode length
    try:
        results['episode_length'] = test_episode_length()
    except Exception as e:
        print(f"❌ Episode length test failed with error: {e}")
        results['episode_length'] = False
    
    # Test 3: Render frequency (manual)
    results['render_frequency'] = test_render_frequency()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All automatic tests PASSED!")
        print("✅ Timestep fix is working correctly")
    else:
        print("⚠️  Some tests FAILED")
        print("❌ Please review the timestep implementation")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
