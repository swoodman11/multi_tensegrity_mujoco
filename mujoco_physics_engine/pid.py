import numpy as np

def debug_print(message, filename="pid.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class PID:
    def __init__(
        self,
        Kp=2.0,
        Ki=0.0,
        Kd=1.0,
        dt=0.01,
        RANGE=[-1.0, 1.0],
        debug_enabled=False,
        smoothing_window: int = 10,
        max_step_change: float | None = 1.0,
        smoothing_mode: str = "moving_average",
    ):
        """PID controller with optional output smoothing.

        Parameters
        ----------
        Kp, Ki, Kd : float
            PID gains.
        dt : float
            Controller timestep (s).
        RANGE : list[float, float]
            Min / max output clamp.
        debug_enabled : bool
            Enable debug prints.
        smoothing_window : int
            Number of previous outputs to include for smoothing. 1 disables smoothing.
        max_step_change : float | None, default=1.0
            Limits |u_t - u_{t-1}| after smoothing (rate limiter). Set to None to disable.
        smoothing_mode : str
            Currently only 'moving_average' supported; reserved for future (e.g. 'ema').
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.RANGE = RANGE
        self.debug_enabled = debug_enabled
        self.last_error = None
        self.cum_error = None
        self.u = np.array([0.0])

        # Smoothing attributes
        self.smoothing_window = max(1, int(smoothing_window))
        self.max_step_change = max_step_change if (max_step_change is None or max_step_change > 0) else None
        self.smoothing_mode = smoothing_mode
        self._u_history: list[float] = []  # store scalar values (assumes scalar control per instance)
    
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
        
        # Update and apply integral term with windup protection
        self.cum_error += error * self.dt
        
        # # Apply integral windup protection
        # if self.integral_limit is not None:
        #     self.cum_error = np.clip(self.cum_error, -self.integral_limit, self.integral_limit)
        
        i_term = self.Ki * self.cum_error
        
        # Calculate derivative term with safety check
        if self.dt > 0:
            d_term = self.Kd * (error - self.last_error) / self.dt
        else:
            d_term = 0.0
        self.last_error = error
        
        # Combine terms (raw control before smoothing)
        raw_u = p_term + i_term + d_term

        # --- Output smoothing ---
        u_smoothed = raw_u
        if self.smoothing_window > 1:
            # Maintain history list length <= smoothing_window-1 (we'll append current raw next)
            if len(self._u_history) >= self.smoothing_window:
                # drop oldest if list already at window length
                self._u_history.pop(0)
            self._u_history.append(float(raw_u))

            if self.smoothing_mode == "moving_average":
                u_smoothed = np.array([np.mean(self._u_history)])
            elif self.smoothing_mode == "ema":  # optional experimental path
                # Use 2/(N+1) smoothing factor approx to match window length
                alpha = 2.0 / (self.smoothing_window + 1.0)
                if len(self._u_history) == 1:
                    u_smoothed = np.array([self._u_history[-1]])
                else:
                    prev = self.u if self.u is not None else np.array([0.0])
                    u_smoothed = alpha * raw_u + (1 - alpha) * prev
            else:
                # Unknown mode -> fallback to raw
                u_smoothed = raw_u
        else:
            self._u_history = [float(raw_u)]  # keep last for possible rate limiting

        # --- Rate limiting (after smoothing) ---
        if self.max_step_change is not None:
            prev_u = self.u if self.u is not None else np.array([0.0])
            delta = u_smoothed - prev_u
            delta = np.clip(delta, -self.max_step_change, self.max_step_change)
            u_smoothed = prev_u + delta

        self.u = u_smoothed
        
        # Safety check for NaN/Inf values
        if not np.isfinite(self.u).all():
            print(f"WARNING: Non-finite PID output detected: {self.u}")
            # self.u = np.zeros_like(self.u)
        
        # Clip control output to range
        self.u = np.clip(self.u, self.RANGE[0], self.RANGE[1])
        
        return self.u
        
    def update_control_by_target_norm_length(self, curr_length, target_norm_length, rest_length, min_length, max_length):
        """
        Update PID controller based on current and target lengths.
        Modified to support bidirectional control (-1.0 to 1.0)
        """
        # Input validation
        if not np.isfinite([curr_length, target_norm_length, rest_length, min_length, max_length]).all():
            print(f"WARNING: Non-finite input to PID controller")
            return np.array([0.0]), None
            
        # Map target_norm_length (0.0-1.0) to desired cable length
        # 0.0 = fully contracted, 1.0 = fully extended
        target_length = min_length + (max_length - min_length) * target_norm_length
        
        # Calculate error (positive error = need to extend, negative error = need to contract)
        error = target_length - curr_length
        
        # # Additional safety check for error magnitude
        # if abs(error) > 2.0:  # Limit maximum error to prevent extreme responses
        #     error = np.sign(error) * 2.0
        
        # Update PID controller
        self.update(error)
        
        # Map output to bidirectional control [-1.0, 1.0]
        # Negative = contract, Positive = extend (opposite of current logic)
        self.u = np.clip(self.u, -1.0, 1.0)
        
        debug_print(
            f"PID: curr={curr_length:.3f}, target={target_length:.3f}, error={error:.3f}, u={self.u}",
            "pid.py",
            self.debug_enabled,
        )
        return self.u, None

    def reset(self):
        self.last_error = None
        self.cum_error = None
        self.done = None
        self._u_history.clear()
