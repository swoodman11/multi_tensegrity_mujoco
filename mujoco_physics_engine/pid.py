import numpy as np

def debug_print(message, filename="pid.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class PID:
    def __init__(self, Kp=2.0, Ki=0.0, Kd=1.0, dt=0.01, RANGE=[-1.0, 1.0], debug_enabled=False):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.RANGE = RANGE
        self.debug_enabled = debug_enabled
        self.last_error = None
        self.cum_error = None
        self.u = np.array([0.0])
    
    def update(self, error):
        """
        Standard PID controller update method.
        """
        # Initialize values on first call
        if self.last_error is None:
            self.last_error = np.array([0.0])
        if self.cum_error is None:
            self.cum_error = np.array([0.0])
            
        # Calculate PID terms
        p_term = self.Kp * error
        
        # Update and apply integral term
        self.cum_error += error * self.dt
        i_term = self.Ki * self.cum_error
        
        # Calculate derivative term
        d_term = self.Kd * (error - self.last_error) / self.dt
        self.last_error = error
        
        # Combine terms
        self.u = p_term + i_term + d_term
        
        # Clip control output to range
        self.u = np.clip(self.u, self.RANGE[0], self.RANGE[1])
        
        return self.u
        
    def update_control_by_target_norm_length(self, curr_length, target_norm_length, rest_length, min_length, max_length):
        """
        Update PID controller based on current and target lengths.
        Modified to support bidirectional control (-1.0 to 1.0)
        """
        # Map target_norm_length (0.0-1.0) to desired cable length
        # 0.0 = fully contracted, 1.0 = fully extended
        # min_length = 0.1  # Same as your min_length clip value
        # max_length = 1.0  # Same as your max_length clip value
        target_length = min_length + (max_length - min_length) * target_norm_length
        # print(f"DEBUG target_length: {target_length}, curr_length: {curr_length}")
        
        # Calculate error (positive error = need to extend, negative error = need to contract)
        error = target_length - curr_length
        
        # Update PID controller
        self.update(error)
        
        # Map output to bidirectional control [-1.0, 1.0]
        # Negative = contract, Positive = extend (opposite of current logic)
        self.u = np.clip(self.u, -1.0, 1.0)
        
        debug_print(f"PID: curr={curr_length:.3f}, target={target_length:.3f}, error={error:.3f}, u={self.u}", "pid.py", self.debug_enabled)
        return self.u, None

    def reset(self):
        self.last_error = None
        self.cum_error = None
        self.done = None
