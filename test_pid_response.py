"""
PID Response Test for Tensegrity Cable Actuators

This script tests the PID controller response for all cable actuators over time,
plotting the response curves and marking the XML timestep intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator

def test_pid_response_step_input(duration_seconds=5.0, target_length=0.9, xml_config="config_2"):
    """
    Test PID response to a step input for all actuators.
    
    Args:
        duration_seconds: Total simulation time
        target_length: Target normalized cable length (0.0 to 1.0)
        xml_config: Which XML configuration to use ("config_1" or "config_2")
    """
    
    # Setup simulator
    if xml_config == "config_1":
        xml_path = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
        xml_timestep = 0.02  # Based on typical config_1 timestep
    else:
        xml_path = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_2.xml')
        xml_timestep = 0.04  # From the XML file
    
    print(f"Testing PID response with {xml_config}")
    print(f"XML timestep: {xml_timestep} seconds")
    print(f"Target length: {target_length} (normalized)")
    print(f"Duration: {duration_seconds} seconds")
    
    # Initialize simulator without visualization for faster testing
    sim = TensegrityMuJoCoSimulator(xml_path, visualize=False, debug_enabled=False)
    
    # Bring robot to ground level and initialize
    sim.bring_to_grnd()
    sim.forward()
    
    # Get initial cable lengths
    initial_lengths = []
    for i in range(sim.n_actuators):
        cable_idx = sim.actuated_ids[i]
        s0 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][0]}").data
        s1 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][1]}").data
        initial_length = np.linalg.norm(s1 - s0)
        initial_lengths.append(initial_length)
    
    print(f"Initial cable lengths: {[f'{l:.3f}' for l in initial_lengths]}")
    
    # Data storage
    num_steps = int(duration_seconds / sim.dt)
    time_array = np.zeros(num_steps)
    cable_lengths = np.zeros((num_steps, sim.n_actuators))
    target_lengths = np.zeros((num_steps, sim.n_actuators))
    control_signals = np.zeros((num_steps, sim.n_actuators))
    
    # Create target length array - step input at t=0
    target_step = [target_length] * sim.n_actuators
    
    print(f"Running simulation for {num_steps} steps...")
    start_time = time.time()
    
    # Run simulation
    for step in range(num_steps):
        # Time tracking
        time_array[step] = step * sim.dt
        
        # Store current cable lengths before step
        for i in range(sim.n_actuators):
            cable_idx = sim.actuated_ids[i]
            s0 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][0]}").data
            s1 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][1]}").data
            current_length = np.linalg.norm(s1 - s0)
            cable_lengths[step, i] = current_length
            target_lengths[step, i] = target_length
        
        # Take simulation step with target lengths
        sim.sim_step(target_step)
        
        # Store control signals after PID update
        for i in range(sim.n_actuators):
            # Prefer direct PID output if available
            if hasattr(sim, 'pid_controllers') and hasattr(sim.pid_controllers[i], 'output'):
                control_signals[step, i] = sim.pid_controllers[i].output  # Accurate PID output
                # print("PID storage option 1")
            elif hasattr(sim, 'curr_ctrl'):
                control_signals[step, i] = sim.curr_ctrl[i]  # Fallback: current control signal
                # print("PID storage option 2")
            else:
                control_signals[step, i] = 0.0  # Default if not available
                # print("PID storage option 3")
            
            # Add debugging prints here
            if step % 25 == 0:  # Print every 25 steps to avoid spam
                # print(f"Step {step}, Actuator {i+1}:")
                # print(f"  Current length: {current_length:.6f}")
                # print(f"  Target length: {target_length:.6f}")
                # print(f"  Error: {target_length - current_length:.6f}")
                # print(f"  Control signal: {control_signals[step, i]:.6f}")
                # Assuming PID components are available
                if hasattr(sim, 'pid_components'):
                    proportional, integral, derivative = sim.pid_components[i]
                    # print(f"  PID components - P: {proportional:.6f}, I: {integral:.6f}, D: {derivative:.6f}")
        
        # Progress indicator
        if step % (num_steps // 10) == 0:
            progress = (step / num_steps) * 100
            # print(f"Progress: {progress:.1f}%")
    
    simulation_time = time.time() - start_time
    # print(f"Simulation completed in {simulation_time:.2f} seconds")
    
    return time_array, cable_lengths, target_lengths, control_signals, xml_timestep, initial_lengths

def plot_pid_response(time_array, cable_lengths, target_lengths, control_signals, xml_timestep, initial_lengths, target_length):
    """
    Plot the PID response for all actuators.
    """
    n_actuators = cable_lengths.shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'PID Response Test - Step Input to {target_length:.1f} (Normalized)', fontsize=16, fontweight='bold')
    
    # Color map for different actuators
    colors = plt.cm.tab20(np.linspace(0, 1, n_actuators))
    
    # Plot 1: Cable Lengths vs Time
    ax1 = axes[0]
    for i in range(n_actuators):
        ax1.plot(time_array, cable_lengths[:, i], color=colors[i], 
                label=f'Actuator {i+1}', linewidth=1.5, alpha=0.8)
        
        # Plot target length as horizontal line
        ax1.axhline(y=target_lengths[0, i], color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Cable Length (m)')
    ax1.set_title('Cable Length Response')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add vertical lines for XML timestep intervals
    max_time = time_array[-1]
    timestep_lines = np.arange(0, max_time + xml_timestep, xml_timestep)
    for t_line in timestep_lines[1:]:  # Skip t=0
        ax1.axvline(x=t_line, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Add timestep annotation
    ax1.text(0.02, 0.98, f'XML timestep: {xml_timestep}s', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Control Signals vs Time
    ax2 = axes[1]
    for i in range(n_actuators):
        ax2.plot(time_array, control_signals[:, i], color=colors[i], 
                label=f'Actuator {i+1}', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Control Signal')
    ax2.set_title('PID Control Signals')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add XML timestep lines
    for t_line in timestep_lines[1:]:
        ax2.axvline(x=t_line, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Plot 3: Error vs Time (Target - Actual)
    ax3 = axes[2]
    for i in range(n_actuators):
        # Convert target normalized length to actual target length for comparison
        min_length = 0.1  # From PID implementation
        max_length = 1.0
        target_actual = min_length + (max_length - min_length) * target_length
        error = target_actual - cable_lengths[:, i]
        ax3.plot(time_array, error, color=colors[i], 
                label=f'Actuator {i+1}', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('PID Error (Target - Actual)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add XML timestep lines
    for t_line in timestep_lines[1:]:
        ax3.axvline(x=t_line, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    
    plt.tight_layout()
    return fig

def test_multiple_step_inputs():
    """
    Test multiple step input values to compare PID responses.
    """
    target_values = [0.2, 0.5, 0.8]  # Different normalized target lengths
    duration = 3.0  # Shorter duration for multiple tests
    
    fig, axes = plt.subplots(len(target_values), 1, figsize=(12, 4*len(target_values)))
    
    for idx, target in enumerate(target_values):
        # print(f"\n--- Testing target length: {target} ---")
        
        time_array, cable_lengths, target_lengths, control_signals, xml_timestep, initial_lengths = \
            test_pid_response_step_input(duration_seconds=duration, target_length=target)
        
        # Plot average response across all actuators
        ax = axes[idx] if len(target_values) > 1 else axes
        
        avg_length = np.mean(cable_lengths, axis=1)
        std_length = np.std(cable_lengths, axis=1)
        
        ax.plot(time_array, avg_length, 'b-', linewidth=2, label='Average Cable Length')
        ax.fill_between(time_array, avg_length - std_length, avg_length + std_length, 
                       alpha=0.3, color='blue', label='±1 Std Dev')
        
        # Target line
        min_length = 0.1
        max_length = 1.0
        target_actual = min_length + (max_length - min_length) * target
        ax.axhline(y=target_actual, color='red', linestyle='--', linewidth=2, label=f'Target ({target:.1f} norm)')
        
        # XML timestep lines
        max_time = time_array[-1]
        timestep_lines = np.arange(0, max_time + xml_timestep, xml_timestep)
        for t_line in timestep_lines[1:]:
            ax.axvline(x=t_line, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cable Length (m)')
        ax.set_title(f'Average PID Response - Target: {target:.1f} (normalized)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add timestep annotation
        ax.text(0.02, 0.98, f'XML timestep: {xml_timestep}s', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to run PID response tests.
    """
    print("=== PID Response Analysis for Tensegrity Cable Actuators ===\n")
    
    try:
        # Test 1: Single step input response
        print("Test 1: Single Step Input Response")
        print("-" * 40)
        
        time_array, cable_lengths, target_lengths, control_signals, xml_timestep, initial_lengths = \
            test_pid_response_step_input(duration_seconds=5.0, target_length=0.9)
        
        # Create and show plot
        fig1 = plot_pid_response(time_array, cable_lengths, target_lengths, control_signals, 
                                xml_timestep, initial_lengths, 0.9)
        
        # Save the plot
        fig1.savefig('pid_response_step_input.png', dpi=300, bbox_inches='tight')
        print("Saved plot: pid_response_step_input.png")
        
        # Test 2: Multiple step inputs comparison
        print("\nTest 2: Multiple Step Input Comparison")
        print("-" * 40)
        
        fig2 = test_multiple_step_inputs()
        fig2.savefig('pid_response_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved plot: pid_response_comparison.png")
        
        # Show plots
        plt.show()
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"XML Timestep: {xml_timestep} seconds")
        print(f"Simulation Timestep: {time_array[1] - time_array[0]:.6f} seconds")
        print(f"Number of Actuators: {cable_lengths.shape[1]}")
        print(f"Initial Cable Lengths: {[f'{l:.3f}m' for l in initial_lengths]}")
        
        # Calculate settling time (time to reach within 5% of target)
        target_actual = 0.1 + (1.0 - 0.1) * 0.7  # For target_length = 0.7
        final_lengths = cable_lengths[-100:, :].mean(axis=0)  # Average of last 100 steps
        
        print(f"Final Cable Lengths: {[f'{l:.3f}m' for l in final_lengths]}")
        print(f"Target Length: {target_actual:.3f}m")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()