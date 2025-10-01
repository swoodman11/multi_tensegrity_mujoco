import numpy as np

def debug_print(message, filename="cable_motor.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class MotorState:
    def __init__(self, sys_precision=np.float64):
        super().__init__()
        self.omega_t = np.zeros(1, dtype=sys_precision)  # angular velocity

    def reset(self):
        self.omega_t = np.zeros(1, dtype=np.float64)


class DCMotor:
    def __init__(self,
                 winch_r=0.035, # unit: meter*10, divide by 10 to get real world value
                 speed=np.array(0.8),
                 sys_precision=np.float64,
                 debug_enabled=False):
        super().__init__()
        self.max_omega = np.array(220 * 2 * np.pi / 60., dtype=sys_precision)
        self.speed = speed
        self.winch_r = winch_r
        self.debug_enabled = debug_enabled
        self.motor_state = MotorState()

    def compute_cable_length_delta(self, ctrl, dt):
        """
        Compute change in cable length based on control signal.
        Modified to support bidirectional control.
        """
        # Convert control signal [-1.0, 1.0] to motor speed
        omega = ctrl * self.max_omega  # Can be positive or negative
        
        # Compute length change (REVERSED: negative dl = shortening, positive dl = extension)
        dl = -omega * self.winch_r * dt  # Note the negative sign to reverse direction
        
        # Debug output
        debug_print(f"Motor: ctrl={ctrl:.3f}, omega={omega:.3f}, dl={dl:.5f}", "cable_motor.py", self.debug_enabled)
        
        return dl

    def reset_omega_t(self):
        self.motor_state.reset()
