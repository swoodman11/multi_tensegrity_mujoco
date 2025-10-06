"""Run a single tensegrity simulation with a JSON action sequence.

Usage examples:

  # Write an example JSON then run it with visualization
  python run_single.py --write-example actions_example.json
  python run_single.py --sequence actions_example.json

  # Headless run saving video & plots
  python run_single.py --sequence actions_example.json --no-vis --video-save --plots

JSON Format:
{
  "actions": [
     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
     [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
  ]
}

Each inner list is a normalized target length vector (6 values) applied for
multiple internal physics integration steps. By design here, a single action
is held for H = round(1 / dt) physics steps (≈ 1.0 simulated second), where
dt is the MuJoCo model timestep. This makes each high-level action represent
approximately one second of simulated time regardless of dt, and total
runtime ≈ num_actions * 1.0 seconds.

CLI Flags:
  --sequence <file>     Path to JSON containing action sequence.
  --write-example <f>   Write an example JSON file and exit.
  --no-vis              Disable viewer.
  --video-save          Save an MP4 of the run (requires --sequence).
  --kp/--ki/--kd        PID gains.
  --total-steps N       Truncate or pad sequence to N steps (repeat last action).
  --plots               Generate plots for actions, observations, rewards.

Action Range Enforcement:
  If an action value falls outside [0,1], the first occurrence raises ValueError.
  Subsequent violations are clipped and counted. A summary prints at the end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
import json

import numpy as np

from mujoco_physics_engine.single_tensegrity_mjc_simulation import (
    SingleTensegrityMuJoCoSimulator,
    load_action_sequence,
    write_example_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run single tensegrity simulation")
    p.add_argument("--sequence", type=Path, help="Path to JSON action sequence", default=None)
    p.add_argument("--write-example", type=Path, help="Write example actions JSON and exit", default=None)
    p.add_argument("--no-vis", action="store_true", help="Run headless (no viewer)")
    p.add_argument("--video-save", action="store_true", help="Save MP4 video (requires visualization enabled internally)")
    p.add_argument("--kp", type=float, default=10.0, help="PID Kp")
    p.add_argument("--ki", type=float, default=0.2, help="PID Ki")
    p.add_argument("--kd", type=float, default=2.0, help="PID Kd")
    p.add_argument("--total-steps", type=int, default=None, help="Optional total steps override")
    p.add_argument("--plots", action="store_true", help="Generate plots for actions/observations/rewards")
    p.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Seconds to sleep after each step when visualizing (default: 0.1, set 0 to disable)."
    )
    p.add_argument(
        "--cam-dist-scale",
        type=float,
        default=9.0,
        help="Scale factor applied once to the initial free camera distance in the interactive viewer (default 9.0)."
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose diagnostic printing (per-step actuator details)."
    )
    p.add_argument(
        "--debug-steps",
        type=int,
        default=20,
        help="Maximum number of initial physics steps to print diagnostics for when --debug is enabled (default 20)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.write_example is not None:
        write_example_json(args.write_example)
        print(f"Wrote example action JSON to {args.write_example}")
        return

    # Determine which action JSON to use
    if args.sequence is None:
        default_path = Path("actions_example.json")
        if not default_path.exists():
            write_example_json(default_path)
            print(f"Generated default action JSON at {default_path}")
        active_actions_path = default_path
        print(f"Using default actions file: {active_actions_path.resolve()}")
    else:
        if not args.sequence.exists():
            print(f"Error: Provided --sequence path does not exist: {args.sequence}", file=sys.stderr)
            sys.exit(1)
        active_actions_path = args.sequence
        print(f"Using provided actions file: {active_actions_path.resolve()}")

    actions_seq = load_action_sequence(active_actions_path)  # shape (T, 6)
    if actions_seq.shape[1] != 6:
        print(f"Error: Expected 6 actions per step, got shape {actions_seq.shape}", file=sys.stderr)
        sys.exit(1)

    # Apply total-steps override (truncate or pad by repeating last action)
    if args.total_steps is not None:
        if args.total_steps <= 0:
            print("Error: --total-steps must be > 0", file=sys.stderr)
            sys.exit(1)
        if args.total_steps < actions_seq.shape[0]:
            actions_seq = actions_seq[: args.total_steps]
        elif args.total_steps > actions_seq.shape[0]:
            last = actions_seq[-1]
            pad = np.repeat(last[None, :], args.total_steps - actions_seq.shape[0], axis=0)
            actions_seq = np.vstack([actions_seq, pad])

    output_dir = Path("./sim_output/single")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always create an internal renderer if we want video, even if --no-vis
    internal_visualize = (not args.no_vis) or args.video_save

    sim = SingleTensegrityMuJoCoSimulator(
        visualize=internal_visualize,
        pid_kp=args.kp,
        pid_ki=args.ki,
        pid_kd=args.kd,
        debug_enabled=args.debug,
        debug_max_steps=args.debug_steps,
    )

    # Determine how many physics steps to hold each high-level action
    hold_steps = max(1, int(round(1.0 / sim.dt)))  # ~1 second per high-level action
    expanded_actions = np.repeat(actions_seq, hold_steps, axis=0)

    print("==== Single Tensegrity Simulation ====")
    print(f"High-level actions: {actions_seq.shape[0]}")
    print(f"Physics dt: {sim.dt:.6f} s  |  Hold steps/action: {hold_steps}  |  Total physics steps: {expanded_actions.shape[0]}")
    print(f"Approx total simulated time: {expanded_actions.shape[0] * sim.dt:.3f} s")
    print("Actuated tendon IDs:", sim.actuated_ids)

    frames = []
    observations = []
    reward_terms_log = []
    rewards = []
    controls_log = []          # PID control outputs per physics step
    lengths_log = []           # Actual cable lengths
    rest_lengths_log = []      # Rest (spring) lengths after motor update

    # Try launching interactive viewer (MuJoCo native window) if visualization enabled
    viewer = None
    if internal_visualize and not args.no_vis:
        try:
            import mujoco.viewer  # type: ignore
            viewer = mujoco.viewer.launch_passive(sim.mjc_model, sim.mjc_data)
            print("[viewer] Interactive MuJoCo viewer launched.")
            # Adjust free camera distance (camera 0 is usually free) if possible
            try:
                if 0 < sim.mjc_model.ncam:
                    # If named camera 'camera' exists we leave it; we adjust the free camera (camid = -1) via scene camera
                    cam = viewer.cam  # mujoco.viewer.Camera object
                    cam.distance *= max(0.1, args.cam_dist_scale)
                    print(f"[viewer] Scaled camera distance by {args.cam_dist_scale:.2f} -> {cam.distance:.3f}")
            except Exception as ce:
                print(f"[viewer] Camera distance scaling failed: {ce}")
        except Exception as e:  # pragma: no cover
            print(f"[viewer] Could not launch interactive viewer: {e}\nProceeding with offscreen rendering only.")

    # Determine effective delay: if user didn't set, use sim.dt when visualizing
    effective_delay = None
    if internal_visualize and not args.no_vis:
        if args.step_delay is None:
            effective_delay = 0.01  # user-requested slower default
        else:
            effective_delay = max(0.0, args.step_delay)

    try:
        prev_high_idx = -1  # Track last printed high-level action index
        for step_i, act in enumerate(expanded_actions):
            high_idx = step_i // hold_steps
            if high_idx != prev_high_idx:
                # Print which high-level action is starting
                print(
                    f"\n>> Executing high-level action {high_idx + 1}/{actions_seq.shape[0]} "
                    f"(holding {hold_steps} physics steps ~{hold_steps * sim.dt:.2f}s): "
                    f"{actions_seq[high_idx].tolist()}"
                )
                prev_high_idx = high_idx
            try:
                obs, reward, done, info = sim.step(act)
            except ValueError as e:
                # First out-of-range violation surfaces here
                print(f"Action error at step {step_i}: {e}", file=sys.stderr)
                # Re-try with clipped action (already clipped internally after raising)
                obs, reward, done, info = sim.step(np.clip(act, 0.0, 1.0))
            observations.append(obs)
            rewards.append(reward)
            reward_terms_log.append(info.get("reward_terms", {}))
            if "controls" in info:
                controls_log.append(info["controls"])
            if "actuated_lengths" in info:
                lengths_log.append(info["actuated_lengths"])
            if "rest_lengths" in info:
                rest_lengths_log.append(info["rest_lengths"])

            # Offscreen frame (only needed for saving video); interactive viewer updates via viewer.sync()
            frame = None
            if args.video_save and internal_visualize:
                frame = sim.render_frame()
            elif args.video_save and not internal_visualize:
                frame = sim.render_frame()
            if frame is not None and args.video_save:
                frames.append(frame)

            if viewer is not None:
                try:
                    viewer.sync()
                except Exception:
                    pass
            if effective_delay and effective_delay > 0:
                time.sleep(effective_delay)

    finally:
        # Summary stats
        print("\n=== Run Summary ===")
        arr_rewards = np.array(rewards)
        print(f"Total physics steps: {len(rewards)} (high-level actions: {actions_seq.shape[0]}; hold/action: {hold_steps})")
        print(f"Reward sum: {arr_rewards.sum():.4f}  mean: {arr_rewards.mean():.4f}  std: {arr_rewards.std():.4f}")
        print(f"Action clip violations: {sim.action_clip_violations}")

        # Video save
        if args.video_save and frames:
            video_path = output_dir / "single_run.mp4"
            sim.save_video(video_path, frames)
            print(f"Saved video to {video_path}")

        # Plots
        if args.plots:
            try:
                import matplotlib.pyplot as plt
                from mujoco_physics_engine.single_tensegrity_mjc_simulation import (
                    SingleTensegrityMuJoCoSimulator as _S,
                )

                # Reuse existing basic plots
                _S.plot_actions(expanded_actions, save_path=output_dir / "actions.png")
                _S.plot_observations(np.asarray(observations), save_path=output_dir / "observations.png")
                _S.plot_rewards(reward_terms_log, save_path=output_dir / "rewards.png")

                # Extended plotting
                t = np.arange(len(rewards)) * sim.dt

                # 1. Controls (PID outputs)
                if controls_log:
                    controls_arr = np.asarray(controls_log)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(t[:controls_arr.shape[0]], controls_arr)
                    ax.set_title("Control Inputs (PID Outputs) vs Time")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Control Signal")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(output_dir / "controls.png", dpi=200)
                    plt.close(fig)

                # 2. Action timeline (expanded)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(t[:len(expanded_actions)], expanded_actions)
                ax.set_title("Applied Normalized Actions vs Time (Expanded)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Action Value (0-1)")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(output_dir / "actions_vs_time.png", dpi=200)
                plt.close(fig)

                # 3. Cable lengths (physical)
                if lengths_log:
                    lengths_arr = np.asarray(lengths_log)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(t[:lengths_arr.shape[0]], lengths_arr)
                    ax.set_title("Cable Lengths vs Time")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Length (m)")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(output_dir / "cable_lengths.png", dpi=200)
                    plt.close(fig)

                # 4. Rest (spring) lengths
                if rest_lengths_log:
                    rest_arr = np.asarray(rest_lengths_log)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(t[:rest_arr.shape[0]], rest_arr)
                    ax.set_title("Rest (Spring) Lengths vs Time")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Rest Length (m)")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(output_dir / "rest_lengths.png", dpi=200)
                    plt.close(fig)

                # 5. Total reward + components
                if reward_terms_log:
                    # Total reward already in rewards list
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(t, rewards, label='total', color='black')
                    # Component curves
                    comp_keys = sorted(reward_terms_log[0].keys())
                    for k in comp_keys:
                        series = [d.get(k,0.0) for d in reward_terms_log]
                        ax.plot(t[:len(series)], series, label=k)
                    ax.set_title("Reward Components vs Time")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Reward")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(output_dir / "reward_components.png", dpi=200)
                    plt.close(fig)

                # Diagnostics plot (target vs actual vs rest + control + error)
                if hasattr(sim, 'diag_target_norm') and sim.diag_target_norm:
                    try:
                        tgt_norm = np.asarray(sim.diag_target_norm)               # (T, n)
                        curr_len = np.asarray(sim.diag_curr_length)              # (T, n)
                        rest_len = np.asarray(sim.diag_rest_length)              # (T, n)
                        pid_u = np.asarray(sim.diag_pid_u)                       # (T, n)
                        err = np.asarray(sim.diag_error)                         # (T, n)
                        n_act = tgt_norm.shape[1]
                        tt = np.arange(tgt_norm.shape[0]) * sim.dt
                        # Convert target norm to physical length
                        tgt_phys = sim.min_cable_length + tgt_norm * (sim.max_cable_length - sim.min_cable_length)

                        cols = n_act
                        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                        # Lengths panel: plot per-actuator actual vs rest vs target (faint lines)
                        for j in range(n_act):
                            axes[0].plot(tt, tgt_phys[:, j], linestyle='--', alpha=0.5)
                        for j in range(n_act):
                            axes[0].plot(tt, rest_len[:, j], linestyle=':', alpha=0.7)
                        for j in range(n_act):
                            axes[0].plot(tt, curr_len[:, j], alpha=0.9)
                        axes[0].set_title('Cable Lengths: target (--) rest (:) actual (solid)')
                        axes[0].set_ylabel('Length (m)')
                        axes[0].grid(alpha=0.3)

                        # Control signals
                        for j in range(n_act):
                            axes[1].plot(tt, pid_u[:, j])
                        axes[1].set_title('PID Control Outputs (u)')
                        axes[1].set_ylabel('u')
                        axes[1].grid(alpha=0.3)

                        # Errors (target - current)
                        for j in range(n_act):
                            axes[2].plot(tt, err[:, j])
                        axes[2].set_title('Length Error (target - current)')
                        axes[2].set_xlabel('Time (s)')
                        axes[2].set_ylabel('Error (m)')
                        axes[2].grid(alpha=0.3)

                        fig.tight_layout()
                        fig.savefig(output_dir / 'diagnostics.png', dpi=200)
                        plt.close(fig)
                    except Exception as de:
                        print(f"Diagnostics plotting failed: {de}")

                print(f"Saved extended plots to {output_dir}")
            except Exception as e:
                print(f"Plotting failed: {e}")

        # Clean up interactive viewer
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
