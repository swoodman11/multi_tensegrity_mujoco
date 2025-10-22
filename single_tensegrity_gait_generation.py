"""Utility to generate gait patterns for single tensegrity robot and save to JSON.

This script provides functions to generate various gait patterns including:
- Single cable activation (test individual cables)
- Alternating tripod patterns
- Wave patterns
- Custom sequences

Run this script to generate new gait JSON files for training.
"""

import json
import numpy as np
from pathlib import Path
from typing import List


def save_gait_to_json(actions: List[List[float]], filename: str, description: str = ""):
    """Save gait sequence to JSON file.
    
    Parameters
    ----------
    actions : list of list of float
        Action sequence, each action is a list of 6 floats in [0, 1]
    filename : str
        Output filename (will be saved in current directory)
    description : str, optional
        Description of the gait pattern
    """
    gait_data = {
        "description": description,
        "robot_type": "single_3bar_tensegrity",
        "num_actuators": 6,
        "sequence_length": len(actions),
        "actions": actions
    }
    
    filepath = Path(filename)
    with open(filepath, 'w') as f:
        json.dump(gait_data, f, indent=2)
    
    print(f"✅ Saved gait to {filepath}")
    print(f"   Sequence length: {len(actions)}")
    print(f"   Description: {description}")


def generate_single_cable_test(cable_idx: int, num_steps: int = 10) -> List[List[float]]:
    """Generate test pattern activating only one cable at a time.
    
    Parameters
    ----------
    cable_idx : int
        Index of cable to activate (0-5)
    num_steps : int, default=10
        Number of steps in sequence
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    for step in range(num_steps):
        action = [0.5] * 6  # Neutral position
        # Vary the target cable between contracted (0.0) and extended (1.0)
        action[cable_idx] = 0.0 if (step // 2) % 2 == 0 else 1.0
        actions.append(action)
    return actions


def generate_all_cables_test(num_steps: int = 60) -> List[List[float]]:
    """Generate test pattern cycling through all cables one at a time.
    
    Parameters
    ----------
    num_steps : int, default=60
        Number of steps in sequence (should be multiple of 6)
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    steps_per_cable = num_steps // 6
    
    for cable_idx in range(6):
        for step in range(steps_per_cable):
            action = [0.5] * 6
            action[cable_idx] = 0.0 if (step // 2) % 2 == 0 else 1.0
            actions.append(action)
    
    return actions


def generate_alternating_tripod(num_cycles: int = 5, steps_per_phase: int = 4) -> List[List[float]]:
    """Generate alternating tripod gait pattern.
    
    Alternates between two sets of 3 cables to create a walking motion.
    
    Parameters
    ----------
    num_cycles : int, default=5
        Number of complete cycles
    steps_per_phase : int, default=4
        Steps per phase transition
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    
    # Define two tripod groups (cables 0,2,4 vs 1,3,5)
    for cycle in range(num_cycles):
        # Phase 1: Group A contracted, Group B extended
        for step in range(steps_per_phase):
            t = step / steps_per_phase  # 0 to 1
            action = [
                0.2 * (1 - t) + 0.8 * t,  # Cable 0: contract to extend
                0.8 * (1 - t) + 0.2 * t,  # Cable 1: extend to contract
                0.2 * (1 - t) + 0.8 * t,  # Cable 2: contract to extend
                0.8 * (1 - t) + 0.2 * t,  # Cable 3: extend to contract
                0.2 * (1 - t) + 0.8 * t,  # Cable 4: contract to extend
                0.8 * (1 - t) + 0.2 * t,  # Cable 5: extend to contract
            ]
            actions.append(action)
        
        # Phase 2: Group B contracted, Group A extended
        for step in range(steps_per_phase):
            t = step / steps_per_phase
            action = [
                0.8 * (1 - t) + 0.2 * t,  # Cable 0: extend to contract
                0.2 * (1 - t) + 0.8 * t,  # Cable 1: contract to extend
                0.8 * (1 - t) + 0.2 * t,  # Cable 2: extend to contract
                0.2 * (1 - t) + 0.8 * t,  # Cable 3: contract to extend
                0.8 * (1 - t) + 0.2 * t,  # Cable 4: extend to contract
                0.2 * (1 - t) + 0.8 * t,  # Cable 5: contract to extend
            ]
            actions.append(action)
    
    return actions


def generate_wave_pattern(num_cycles: int = 3, steps_per_wave: int = 18) -> List[List[float]]:
    """Generate wave pattern that propagates through cables.
    
    Parameters
    ----------
    num_cycles : int, default=3
        Number of wave cycles
    steps_per_wave : int, default=18
        Steps per complete wave (should be multiple of 6)
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    
    for cycle in range(num_cycles):
        for step in range(steps_per_wave):
            action = []
            for cable_idx in range(6):
                # Create sinusoidal wave with phase offset for each cable
                phase = 2 * np.pi * (step / steps_per_wave + cable_idx / 6)
                value = 0.5 + 0.4 * np.sin(phase)  # Range [0.1, 0.9]
                action.append(float(value))
            actions.append(action)
    
    return actions


def generate_sequential_contraction(num_cycles: int = 3) -> List[List[float]]:
    """Generate pattern where cables contract sequentially in order.
    
    Parameters
    ----------
    num_cycles : int, default=3
        Number of complete cycles through all cables
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    
    for cycle in range(num_cycles):
        for active_cable in range(6):
            # All cables extended except the active one
            action = [0.9] * 6
            action[active_cable] = 0.1
            actions.append(action)
            
            # Hold contracted for a step
            actions.append(action.copy())
            
            # Transition back to neutral
            action_neutral = [0.5] * 6
            action_neutral[active_cable] = 0.1
            actions.append(action_neutral)
    
    return actions


def generate_pulsing_pattern(num_pulses: int = 10, pulse_duration: int = 4) -> List[List[float]]:
    """Generate pattern where all cables pulse together.
    
    Parameters
    ----------
    num_pulses : int, default=10
        Number of pulses
    pulse_duration : int, default=4
        Steps per pulse cycle
    
    Returns
    -------
    actions : list of list of float
    """
    actions = []
    
    for pulse in range(num_pulses):
        for step in range(pulse_duration):
            t = step / pulse_duration
            # All cables go from extended to contracted together
            value = 0.8 * (1 - t) + 0.2 * t
            action = [value] * 6
            actions.append(action)
        
        for step in range(pulse_duration):
            t = step / pulse_duration
            # All cables go from contracted to extended together
            value = 0.2 * (1 - t) + 0.8 * t
            action = [value] * 6
            actions.append(action)
    
    return actions


def generate_random_walk(num_steps: int = 50, step_size: float = 0.2, seed: int = 42) -> List[List[float]]:
    """Generate random walk pattern for exploration.
    
    Parameters
    ----------
    num_steps : int, default=50
        Number of steps
    step_size : float, default=0.2
        Maximum change per step
    seed : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    actions : list of list of float
    """
    np.random.seed(seed)
    actions = []
    
    # Start from neutral position
    current = np.array([0.5] * 6)
    
    for step in range(num_steps):
        # Random walk with clipping
        delta = np.random.uniform(-step_size, step_size, 6)
        current = np.clip(current + delta, 0.0, 1.0)
        actions.append(current.tolist())
    
    return actions


def main():
    """Generate and save various gait patterns."""
    print("🤖 Single Tensegrity Gait Pattern Generator")
    print("=" * 60)
    
    # 1. Single cable tests (one file per cable)
    print("\n📝 Generating single cable test patterns...")
    for cable_idx in range(6):
        actions = generate_single_cable_test(cable_idx, num_steps=20)
        save_gait_to_json(
            actions, 
            f"test_cable_{cable_idx}.json",
            f"Test pattern for cable {cable_idx} - alternates between contracted and extended"
        )
    
    # 2. All cables cycling test
    print("\n📝 Generating all cables cycling test...")
    actions = generate_all_cables_test(num_steps=60)
    save_gait_to_json(
        actions,
        "test_all_cables.json",
        "Cycles through all cables one at a time for testing"
    )
    
    # 3. Alternating tripod pattern
    print("\n📝 Generating alternating tripod pattern...")
    actions = generate_alternating_tripod(num_cycles=5, steps_per_phase=4)
    save_gait_to_json(
        actions,
        "alternating_tripod.json",
        "Alternating tripod gait - cables 0,2,4 vs 1,3,5"
    )
    
    # 4. Wave pattern
    print("\n📝 Generating wave pattern...")
    actions = generate_wave_pattern(num_cycles=3, steps_per_wave=18)
    save_gait_to_json(
        actions,
        "wave_pattern.json",
        "Sinusoidal wave propagating through cables"
    )
    
    # 5. Sequential contraction
    print("\n📝 Generating sequential contraction pattern...")
    actions = generate_sequential_contraction(num_cycles=3)
    save_gait_to_json(
        actions,
        "sequential_contraction.json",
        "Cables contract one at a time in sequence"
    )
    
    # 6. Pulsing pattern
    print("\n📝 Generating pulsing pattern...")
    actions = generate_pulsing_pattern(num_pulses=10, pulse_duration=4)
    save_gait_to_json(
        actions,
        "pulsing_pattern.json",
        "All cables pulse together in synchrony"
    )
    
    # 7. Random walk
    print("\n📝 Generating random walk pattern...")
    actions = generate_random_walk(num_steps=50, step_size=0.15, seed=42)
    save_gait_to_json(
        actions,
        "random_walk.json",
        "Random walk exploration pattern"
    )
    
    print("\n" + "=" * 60)
    print("✅ All gait patterns generated successfully!")
    print("\nGenerated files:")
    print("  - test_cable_0.json through test_cable_5.json")
    print("  - test_all_cables.json")
    print("  - alternating_tripod.json")
    print("  - wave_pattern.json")
    print("  - sequential_contraction.json")
    print("  - pulsing_pattern.json")
    print("  - random_walk.json")
    print("\nUse these files with gpu_pretraining_SAC_single.py")


if __name__ == "__main__":
    main()
