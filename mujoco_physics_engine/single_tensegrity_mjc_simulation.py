"""Single tensegrity MuJoCo simulator.

This module provides a light-weight, modular version of the multi-tensegrity
simulator focused on a single 3-bar tensegrity robot described by
`xml_models/3bar_new_platform_all_cables.xml`.

Key design goals:
 - Only 6 actuated (active) cables: those with stiffness=1000 in the XML
 - Remaining (passive) cables with stiffness=20000 are left un-actuated
 - Actions are normalized target cable lengths in [0,1]
 - Modular observation pipeline (components can be toggled)
 - Modular reward pipeline (terms exposed & logged)
 - Minimal external dependencies (reuse existing PID, DCMotor, AbstractMuJoCoSimulator)
 - Explicit plotting utilities for actions, observations, and reward terms
 - Easily extensible for RL integration (Gymnasium wrapper can call reset()/step())

NOTE ON SITE NAMES:
The original dual-robot simulator used prefixed site names (t1_/t2_). The
single XML (`3bar_new_platform_all_cables.xml`) does NOT use these prefixes.
We therefore map the first six stiffness=1000 tendons to actuators and build
their site pairs directly by reading tendon definitions.

If you later update the XML naming convention, adjust `_discover_cable_sites`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import mujoco

from .mujoco_simulation import AbstractMuJoCoSimulator
from .pid import PID
from .cable_motor import DCMotor


def debug_print(msg: str, enabled: bool = False):
    if enabled:
        print(f"[single_sim] {msg}")


@dataclass
class ObservationConfig:
    include_imu: bool = True
    include_prev_action: bool = True
    include_cable_lengths: bool = True
    include_cable_length_rates: bool = True


@dataclass
class RewardTerms:
    # Each term returns float
    terms: Dict[str, float] = field(default_factory=dict)

    def total(self) -> float:
        return float(sum(self.terms.values()))

    def add(self, name: str, value: float):
        self.terms[name] = float(value)


class SingleTensegrityMuJoCoSimulator(AbstractMuJoCoSimulator):
    """Single tensegrity simulator with 6 actuated cables.

    Action: np.ndarray shape (6,) normalized target lengths in [0,1].
    """

    def __init__(
        self,
        xml_path: Path | str = Path("mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml"),
        visualize: bool = True,
        render_size: Tuple[int, int] = (720, 720),
        render_fps: int = 20,
        pid_kp: float = 2.0,
        pid_ki: float = 0.0,
        pid_kd: float = 1.0,
        min_cable_length: float = 0.6,
        max_cable_length: float = 1.6,
        debug_enabled: bool = False,
        debug_max_steps: int = 20,
        obs_config: Optional[ObservationConfig] = None,
    ):
        super().__init__(Path(xml_path), visualize, render_size, render_fps)
        self.debug_enabled = debug_enabled
        self.debug_max_steps = max(0, int(debug_max_steps))
        self.min_cable_length = min_cable_length
        self.max_cable_length = max_cable_length
        self.obs_config = obs_config or ObservationConfig()

        # Discover cable sites & stiffness to identify actuated vs passive
        self.cable_sites: List[Tuple[str, str]] = []  # index -> (siteA, siteB)
        self.cable_stiffness: List[float] = []       # index -> stiffness
        self._discover_cable_sites()

        # Active (actuated) are those with stiffness approx 1000.0
        self.actuated_ids = [i for i, k in enumerate(self.cable_stiffness) if abs(k - 1000.0) < 1e-3][:6]
        if len(self.actuated_ids) < 6:
            debug_print("WARNING: Fewer than 6 stiffness=1000 tendons found; check XML.", True)
        self.n_actuators = len(self.actuated_ids)

        # PID and motor for each actuator
        self.pids: List[PID] = [PID(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd, dt=self.dt, debug_enabled=debug_enabled) for _ in range(self.n_actuators)]
        self.cable_motors: List[DCMotor] = [DCMotor(debug_enabled=debug_enabled) for _ in range(self.n_actuators)]

        # Buffers
        self.prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.prev_lengths = np.zeros(self.n_actuators, dtype=np.float32)
        self.action_clip_violations = 0
        self.first_clip_exception_raised = False
        # Instrumentation buffers (grow per step)
        self.diag_target_norm = []  # list of np.ndarray
        self.diag_curr_length = []
        self.diag_rest_length = []
        self.diag_pid_u = []
        self.diag_error = []
        # Extra diagnostics
        self.diag_true_tendon_length = []  # MuJoCo internal tendon length (path length)
        self.diag_dl = []  # motor-applied delta rest length each step

        # Build IMU geometry assumptions (use rod geoms present in single XML: r01, r23, r45)
        self.imu_geom_names = ["r01", "r23", "r45"]

        # Initialize
        self.reset()
        # Internal flag to disable nested component debug after threshold
        self._child_debug_disabled = False

    # ---------------------------------------------------------------------
    # Model Introspection
    # ---------------------------------------------------------------------
    def _discover_cable_sites(self):
        """Populate cable_sites and cable_stiffness from model tendon definitions.

        For each spatial tendon, we extract the two site names. Single XML uses
        names td_0 .. etc; we map their site references via mjModel.tendon_adr
        and wrap references (2 sites per spatial tendon assumed).
        """
        model = self.mjc_model
        for t_idx in range(model.ntendon):
            # Stiffness from model.tendon_stiffness[t_idx]
            stiff = float(model.tendon_stiffness[t_idx])
            self.cable_stiffness.append(stiff)

            # Sites in this tendon span: use the EFC address arrays
            # Access via model.tendon_adr & model.wrap_obj / wrap_type sequence.
            adr = model.tendon_adr[t_idx]
            # For spatial tendon: sequence of site ids. We'll collect first & last site.
            nwrap = model.tendon_num[t_idx]
            site_ids = []
            for w in range(nwrap):
                obj_type = model.wrap_type[adr + w]
                obj_id = model.wrap_objid[adr + w]
                # 0 == site for mjtWrap (per MuJoCo docs)
                if obj_type == mujoco.mjtWrap.mjWRAP_SITE:
                    site_ids.append(obj_id)
            if len(site_ids) >= 2:
                sA = model.site(site_ids[0]).name
                sB = model.site(site_ids[-1]).name
                self.cable_sites.append((sA, sB))
            else:
                self.cable_sites.append(("", ""))

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def reset(self):  # noqa: D401
        super().reset()
        self.forward()
        # Recompute lengths
        self.prev_lengths = self._get_actuated_lengths()
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.action_clip_violations = 0
        self.first_clip_exception_raised = False
        return self.get_observation()

    def step(self, action: np.ndarray):
        """Apply one action (normalized target lengths) then advance one physics step.

        Action shape must match n_actuators. Each element in [0,1].
        Length mapping: target_length = min + a*(max-min)
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_actuators,):
            raise ValueError(f"Action shape {action.shape} != ({self.n_actuators},)")

        # Disable underlying PID/DCMotor debug after configured number of steps (without modifying their source files)
        if (self.debug_enabled and not self._child_debug_disabled and self.step_count >= self.debug_max_steps):
            for pid in self.pids:
                pid.debug_enabled = False
            for mot in self.cable_motors:
                mot.debug_enabled = False
            self._child_debug_disabled = True
            print(f"[debug] Disabled PID/DCMotor internal debug after {self.step_count} steps (limit {self.debug_max_steps}).")

        # Clip & track violations
        clipped = np.clip(action, 0.0, 1.0)
        if not np.allclose(clipped, action):
            self.action_clip_violations += 1
            if not self.first_clip_exception_raised:
                self.first_clip_exception_raised = True
                raise ValueError("Action out of [0,1] range encountered. Subsequent violations will be counted only.")
            action = clipped
        else:
            action = clipped

        # Compute controls via PID for each actuator (PID output treated as motor command)
        controls = np.zeros(self.n_actuators, dtype=np.float32)
        target_lengths_m = np.zeros(self.n_actuators, dtype=np.float32)
        current_lengths_m = np.zeros(self.n_actuators, dtype=np.float32)
        errors_m = np.zeros(self.n_actuators, dtype=np.float32)
        pid_us = np.zeros(self.n_actuators, dtype=np.float32)

        for idx, tendon_id in enumerate(self.actuated_ids):
            target_norm = action[idx]
            curr_len = self._tendon_current_length(tendon_id)
            rest_len = self.mjc_model.tendon_lengthspring[tendon_id, 0]
            # Map target_norm to physical target length for diagnostics
            tgt_len = self.min_cable_length + target_norm * (self.max_cable_length - self.min_cable_length)
            u, _ = self.pids[idx].update_control_by_target_norm_length(
                curr_len, target_norm, rest_len, self.min_cable_length, self.max_cable_length
            )
            # Reintroduce inversion so that negative PID (contract) leads to rest-length decrease
            controls[idx] = float(-u)
            target_lengths_m[idx] = tgt_len
            current_lengths_m[idx] = curr_len
            errors_m[idx] = tgt_len - curr_len
            pid_us[idx] = float(u)

        # Update rest lengths using motor dynamics
        for idx, tendon_id in enumerate(self.actuated_ids):
            ctrl = controls[idx]
            rest_length = self.mjc_model.tendon_lengthspring[tendon_id, 0]
            dl = self.cable_motors[idx].compute_cable_length_delta(ctrl, self.dt)
            new_rest = np.clip(rest_length + dl, self.min_cable_length, self.max_cable_length)
            # Fix 4: consistent indexing, always assign [tendon_id, 0]
            self.mjc_model.tendon_lengthspring[tendon_id, 0] = new_rest
            if self.debug_enabled and self.step_count < self.debug_max_steps:
                print(f"[diag] step={self.step_count} act={idx} tendon={tendon_id} target_norm={action[idx]:.3f} tgt_len={target_lengths_m[idx]:.4f} curr_len_pre={current_lengths_m[idx]:.4f} rest_pre={rest_length:.4f} u={pid_us[idx]:.3f} ctrl={ctrl:.3f} dl={dl:.6f} rest_new={new_rest:.4f}")

        mujoco.mj_step(self.mjc_model, self.mjc_data)
        self.forward()
        self.step_count += 1

        obs = self.get_observation(action=action)
        reward, reward_terms = self.compute_reward()
        # Capture current actuator state post-step
        curr_lengths = self._get_actuated_lengths()
        curr_rest_lengths = np.array([
            self.mjc_model.tendon_lengthspring[t_id, 0] for t_id in self.actuated_ids
        ], dtype=np.float32)
        true_tendon_lengths = np.array([
            self.mjc_data.ten_length[t_id] for t_id in self.actuated_ids
        ], dtype=np.float32)
        done = False
        # Instrumentation (Fix 6)
        self.diag_target_norm.append(action.copy())
        self.diag_curr_length.append(current_lengths_m.copy())
        self.diag_rest_length.append(curr_rest_lengths.copy())
        self.diag_pid_u.append(pid_us.copy())
        self.diag_error.append(errors_m.copy())
        self.diag_true_tendon_length.append(true_tendon_lengths.copy())
        # For dl we only stored last loop; recompute quickly here for storage based on rest change
        # (approx) dl = rest_new - rest_old per actuator
        # Since we didn't cache rest_old per actuator collectively, skip exact; placeholder zeros array
        self.diag_dl.append(np.zeros_like(curr_rest_lengths))  # optional refinement later

        if self.debug_enabled and self.step_count <= self.debug_max_steps:
            for idx, tendon_id in enumerate(self.actuated_ids):
                print(
                    f"[diag-post] step={self.step_count} act={idx} tendon={tendon_id} curr_len_post={curr_lengths[idx]:.4f} true_len={true_tendon_lengths[idx]:.4f} rest_len={curr_rest_lengths[idx]:.4f} error={errors_m[idx]:.4f}"
                )

        info = {
            "reward_terms": reward_terms.terms,
            "controls": controls.copy(),
            "actuated_lengths": curr_lengths.copy(),
            "rest_lengths": curr_rest_lengths.copy(),
            "true_tendon_lengths": true_tendon_lengths.copy(),
            "diag_available": True,
        }
        self.prev_action = action.copy()
        self.prev_lengths = self._get_actuated_lengths()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Observation & Reward
    # ------------------------------------------------------------------
    def _tendon_current_length(self, tendon_id: int) -> float:
        # Approximate length from the site endpoints we cached
        try:
            sA, sB = self.cable_sites[tendon_id]
            if not sA or not sB:
                return 0.0
            pA = self.mjc_data.site(self.mjc_model.site(sA).id).xpos
            pB = self.mjc_data.site(self.mjc_model.site(sB).id).xpos
            return float(np.linalg.norm(pB - pA))
        except Exception:
            return 0.0

    def _get_actuated_lengths(self) -> np.ndarray:
        return np.array([self._tendon_current_length(tid) for tid in self.actuated_ids], dtype=np.float32)

    def _get_actuated_length_rates(self) -> np.ndarray:
        curr = self._get_actuated_lengths()
        rates = (curr - self.prev_lengths) / max(self.dt, 1e-8)
        return rates.astype(np.float32)

    def _get_IMU_grav_vectors(self) -> np.ndarray:
        # For each rod geom, approximate orientation via body quaternion
        vecs = []
        gravity_world = np.array([0, 0, -1.0], dtype=np.float32)
        for g in self.imu_geom_names:
            try:
                body_id = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_BODY, g)
                quat = self.mjc_data.xquat[body_id]
                # Convert quaternion to rotation matrix
                w, x, y, z = quat
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
                ], dtype=np.float32)
                vecs.append((R.T @ gravity_world).astype(np.float32))
            except Exception:
                vecs.append(np.zeros(3, dtype=np.float32))
        return np.concatenate(vecs).astype(np.float32)

    def get_observation(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.obs_config.include_cable_lengths:
            lengths = self._get_actuated_lengths()
            norm = (lengths - self.min_cable_length) / max(self.max_cable_length - self.min_cable_length, 1e-6)
            norm = np.clip(norm, 0.0, 1.0)
            parts.append(norm.astype(np.float32))
        if self.obs_config.include_cable_length_rates:
            rates = self._get_actuated_length_rates()
            # Scale rates by length range
            denom = max(self.max_cable_length - self.min_cable_length, 1e-6)
            rates_norm = np.clip(rates / denom, -1.0, 1.0).astype(np.float32)
            parts.append(rates_norm)
        if self.obs_config.include_prev_action:
            parts.append(self.prev_action.astype(np.float32))
        if self.obs_config.include_imu:
            parts.append(self._get_IMU_grav_vectors())
        obs = np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, dtype=np.float32)
        return obs

    def compute_reward(self) -> Tuple[float, RewardTerms]:
        terms = RewardTerms()
        # Example base term: encourage change in average cable length (activity)
        lengths = self._get_actuated_lengths()
        activity = float(np.mean(np.abs(lengths - self.prev_lengths)))
        terms.add("activity", activity)
        # Placeholder for future domain-specific terms
        return terms.total(), terms

    # ------------------------------------------------------------------
    # Plotting Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def plot_actions(actions: np.ndarray, save_path: Optional[Path] = None):
        import matplotlib.pyplot as plt
        actions = np.asarray(actions)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(actions)
        axes[0].set_title("Normalized Actions (Target Lengths)")
        axes[0].set_ylabel("Norm Length")
        # Denormalized example (requires assumed min/max = 0.6 / 1.6)
        denorm = 0.6 + actions * (1.6 - 0.6)
        axes[1].plot(denorm)
        axes[1].set_title("Denormalized Target Lengths")
        axes[1].set_ylabel("Length (m*10)")
        axes[1].set_xlabel("Step")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        return fig

    @staticmethod
    def plot_rewards(reward_terms_list: List[Dict[str, float]], save_path: Optional[Path] = None):
        import matplotlib.pyplot as plt
        if not reward_terms_list:
            return None
        keys = sorted(reward_terms_list[0].keys())
        fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(10, 2 * (len(keys) + 1)), sharex=True)
        totals = []
        for k_i, k in enumerate(keys):
            series = [d.get(k, 0.0) for d in reward_terms_list]
            axes[k_i].plot(series)
            axes[k_i].set_ylabel(k)
            totals.append(series)
        # overall
        overall = [sum(d.values()) for d in reward_terms_list]
        axes[-1].plot(overall, color='black')
        axes[-1].set_ylabel('total')
        axes[-1].set_xlabel('Step')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        return fig

    @staticmethod
    def plot_observations(observations: np.ndarray, save_path: Optional[Path] = None):
        import matplotlib.pyplot as plt
        observations = np.asarray(observations)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(observations.T, aspect='auto', interpolation='nearest')
        ax.set_title('Observations Over Time')
        ax.set_ylabel('Feature Index')
        ax.set_xlabel('Step')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        return fig


def load_action_sequence(json_path: Path) -> np.ndarray:
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'actions' in data:
        seq = np.asarray(data['actions'], dtype=np.float32)
    else:
        seq = np.asarray(data, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    return seq


EXAMPLE_ACTIONS_JSON = {
    "actions": [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ]
}


def write_example_json(path: Path):
    with open(path, 'w') as f:
        json.dump(EXAMPLE_ACTIONS_JSON, f, indent=2)


__all__ = [
    "SingleTensegrityMuJoCoSimulator",
    "ObservationConfig",
    "RewardTerms",
    "load_action_sequence",
    "write_example_json",
]
