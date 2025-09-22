import numpy as np


class MotorState:
    def __init__(self, sys_precision=np.float64):
        super().__init__()
        self.omega_t = np.zeros(1, dtype=sys_precision)  # angular velocity

    def reset(self):
        self.omega_t = np.zeros(1, dtype=np.float64)


class DCMotor:
    def __init__(self,
                 winch_r=0.035,
                 speed=np.array(0.8),
                 sys_precision=np.float64):
        super().__init__()
        self.max_omega = np.array(220 * 2 * np.pi / 60., dtype=sys_precision)
        self.speed = speed
        self.winch_r = winch_r
        self.motor_state = MotorState()

    def compute_cable_length_delta(self, control, delta_t, dim_scale=1.):
        pre_omega = self.motor_state.omega_t.copy()
        self.motor_state.omega_t = np.array(self.speed * self.max_omega * control).reshape(-1)
        dl = (pre_omega + self.motor_state.omega_t) / 2. * self.winch_r * dim_scale * delta_t

        return dl

    def reset_omega_t(self):
        self.motor_state.reset()
