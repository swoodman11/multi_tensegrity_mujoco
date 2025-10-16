"""
Compare reward and penalty components across one or more action sequences.

- Loads JSON sequences containing normalized target lengths in [0,1].
- Runs TensegrityMuJoCoSimulator directly (no Gym env), resetting between sequences.
- For each sequence step, holds each high-level action for H physics steps computed from dt
  (default H = round(1.0 / dt) ≈ one second per high-level action).
- Collects per-physics-step: observation (96D tier2 by default), action (12), control signals (12),
  reward total and each individual component in info.
- Plots:
  - Observations grouped by category across the 96D vector (Tier-2 mode)
  - Control actions (12) over time
  - Total reward and individual reward/penalty components over time

No files are saved; everything is displayed interactively.

Usage examples:
  python test_reward_comp.py --sequences action_sequences/hand_designed_double_gait.json
  python test_reward_comp.py --sequences action_sequences/baseline_single_cable.json hand_designed_double_gait.json
  python test_reward_comp.py --xaxis seconds  # switch plots to seconds

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator


DEFAULT_XML = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')


def load_sequence(path: Path) -> np.ndarray:
    """Load a sequence JSON into a (T, 12) NumPy array of floats in [0,1].
    Accepts two schemas:
      - plain JSON array [[..12..], [..12..], ...]
      - object with key "actions": {"actions": [[..], ...]}
    """
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'actions' in data:
        arr = np.asarray(data['actions'], dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected (T,12) array in {path}, got shape {arr.shape}")
    # Sanity clip to [0,1]
    return np.clip(arr, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Reward/penalty component comparison tool")
    p.add_argument('--sequences', type=Path, nargs='+', required=True,
                   help='List of JSON files with action sequences (normalized [0,1], 12 dims)')
    p.add_argument('--xml', type=Path, default=DEFAULT_XML, help='XML model path to load')
    p.add_argument('--visualize', action='store_true', help='Enable MuJoCo viewer (off by default)')
    p.add_argument('--hold-seconds', type=float, default=1.0,
                   help='Approx seconds to hold each high-level action (default 1.0s)')
    p.add_argument('--xaxis', choices=['steps', 'seconds'], default='steps',
                   help='X-axis units for plots')
    p.add_argument('--kp', type=float, default=10.0)
    p.add_argument('--ki', type=float, default=0.2)
    p.add_argument('--kd', type=float, default=2.0)
    return p.parse_args()


def run_sequence(sim: TensegrityMuJoCoSimulator, actions_seq: np.ndarray, hold_steps: int) -> Dict[str, Any]:
    """Execute a sequence in the simulator and collect in-memory logs per physics step.

    Returns a dict with keys: 'obs', 'actions', 'controls', 'rewards', 'reward_terms', 'times'.
    """
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    ctrl_list: List[np.ndarray] = []
    rew_list: List[float] = []
    terms_list: List[Dict[str, float]] = []

    expanded_actions = np.repeat(actions_seq, hold_steps, axis=0)
    t = np.arange(expanded_actions.shape[0]) * sim.dt

    for a in expanded_actions:
        # Direct step through simulator using normalized target lengths
        o, r, d, info = sim.sim_step(a)
        obs_list.append(o)
        act_list.append(info.get('action', np.asarray(a, dtype=float)))
        if 'controls' in info and info['controls'] is not None:
            ctrl_list.append(np.asarray(info['controls'], dtype=float))
        else:
            ctrl_list.append(np.zeros(sim.n_actuators, dtype=float))
        rew_list.append(float(r))

        # Flatten reward/penalty components (names come from simulator info)
        comp = {}
        if 'reward_components' in info:
            comp.update({k: float(v) for k, v in info['reward_components'].items()})
        if 'penalty_components' in info:
            comp.update({k: float(v) for k, v in info['penalty_components'].items()})
        if 'control_penalty_total' in info:
            comp['control_penalty_total'] = float(info['control_penalty_total'])
        terms_list.append(comp)

    return {
        'obs': np.asarray(obs_list, dtype=float),
        'actions': np.asarray(act_list, dtype=float),
        'controls': np.asarray(ctrl_list, dtype=float),
        'rewards': np.asarray(rew_list, dtype=float),
        'reward_terms': terms_list,
        'times': t,
    }


def plot_observations_tier2(obs: np.ndarray, times: np.ndarray, xaxis: str, title: str):
    """Plot the 96-D Tier-2 observation grouped by known segments.
    Groups:
      0-11: cable_lengths_norm (12)
      12-23: cable_rates_norm (12)
      24-35: prev_action (12)
      36-53: strain_exts (18)
      54-71: imu_grav (18)
      72-89: imu_ang_norm (18)
      90-92: com_lin_vel_norm (3)
      93-95: com_ang_vel_norm (3)
    """
    assert obs.shape[1] >= 96, f"Expected obs dim >=96, got {obs.shape[1]}"
    t = np.arange(obs.shape[0]) if xaxis == 'steps' else times

    groups = [
        (slice(0, 12), 'cable_len_norm'),
        (slice(12, 24), 'cable_rate_norm'),
        (slice(24, 36), 'prev_action'),
        (slice(36, 54), 'strain_ext'),
        (slice(54, 72), 'imu_grav'),
        (slice(72, 90), 'imu_ang_norm'),
        (slice(90, 93), 'com_lin_vel'),
        (slice(93, 96), 'com_ang_vel'),
    ]

    n_rows = 4
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Observations (Tier-2) - {title}")

    for ax, (sl, name) in zip(axes.ravel(), groups):
        ax.plot(t, obs[:, sl])
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel('Time (steps)' if xaxis == 'steps' else 'Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)


essential_components = [
    'cumulative_rotation_reward',
    'consistent_direction_reward',
    'displacement_progress_reward',
    'velocity_reward',
    'distance_reward',
    'control_penalty_total',
]


def plot_rewards(terms_list: List[Dict[str, float]], rewards: np.ndarray, times: np.ndarray, xaxis: str, title: str):
    t = np.arange(len(rewards)) if xaxis == 'steps' else times

    # Collect keys
    all_keys = set()
    for d in terms_list:
        all_keys.update(d.keys())
    # Ensure essential ones are present in plotting order
    ordered = [k for k in essential_components if k in all_keys]
    others = sorted([k for k in all_keys if k not in ordered])
    keys = ordered + others

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, rewards, label='total_reward', linewidth=2, color='black')
    for k in keys:
        vals = np.array([d.get(k, 0.0) for d in terms_list], dtype=float)
        ax.plot(t, vals, label=k, alpha=0.8)
    ax.set_title(f"Rewards and Components - {title}")
    ax.set_xlabel('Time (steps)' if xaxis == 'steps' else 'Time (s)')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show(block=False)


def plot_actions(actions: np.ndarray, controls: np.ndarray, times: np.ndarray, xaxis: str, title: str):
    t = np.arange(actions.shape[0]) if xaxis == 'steps' else times

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(t, actions)
    ax1.set_title('Normalized Target Lengths (Actions)')
    ax1.set_ylabel('Action [0-1]')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, controls)
    ax2.set_title('Control Signals (PID outputs) [-1,1]')
    ax2.set_xlabel('Time (steps)' if xaxis == 'steps' else 'Time (s)')
    ax2.set_ylabel('Control')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show(block=False)


def main():
    args = parse_args()

    # Build simulator per user request (visualization off by default)
    sim = TensegrityMuJoCoSimulator(
        xml_path=args.xml,
        visualize=bool(args.visualize),
        obs_dim=96,
        obs_mode='tier2',
        controller_kp=args.kp,
        controller_ki=args.ki,
        controller_kd=args.kd,
    )

    # Fully reset between sequences (fresh instance as requested)
    def fresh_sim() -> TensegrityMuJoCoSimulator:
        return TensegrityMuJoCoSimulator(
            xml_path=args.xml,
            visualize=bool(args.visualize),
            obs_dim=96,
            obs_mode='tier2',
            controller_kp=args.kp,
            controller_ki=args.ki,
            controller_kd=args.kd,
        )

    # Determine per-action hold steps from dt and requested seconds
    hold_steps = max(1, int(round(args.hold_seconds / sim.dt)))

    for seq_path in args.sequences:
        actions_seq = load_sequence(seq_path)
        # Fresh simulator instance per sequence
        sim = fresh_sim()
        sim.reset()
        sim.bring_to_grnd()

        logs = run_sequence(sim, actions_seq, hold_steps)

        title = f"{Path(seq_path).name} (H={hold_steps} steps/action, dt={sim.dt:.4f}s)"
        plot_observations_tier2(logs['obs'], logs['times'], args.xaxis, title)
        plot_actions(logs['actions'], logs['controls'], logs['times'], args.xaxis, title)
        plot_rewards(logs['reward_terms'], logs['rewards'], logs['times'], args.xaxis, title)

    # Keep figures open until closed by user
    print("Close the plot windows to finish.")
    plt.show()


if __name__ == '__main__':
    main()
