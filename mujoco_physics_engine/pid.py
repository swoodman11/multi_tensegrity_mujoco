import numpy as np


class PID:
    def __init__(self,
                 k_p=6.0,
                 k_i=0.01,
                 k_d=0.5,
                 min_length=60.0, # unit: mm
                 RANGE=100.0, # unit: mm
                 tol=0.1,
                 sys_precision=np.float64):
        self.sys_precision = sys_precision
        self.last_error = None
        self.cum_error = None
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.min_length = min_length/100.0 # convert to meter
        self.RANGE = RANGE/100.0 # convert to meter
        self.tol = tol
        self.LEFT_RANGE = None
        self.RIGHT_RANGE = None
        self.done = None

        # print("Min length:", self.min_length)
        # print("Range:", self.RANGE)

    def update_control_by_target_norm_length(self, current_length, target_norm_length, rest_length):
        """
        :param current_length: absolute length of cable
        :param target_norm_length: normalized target length between 0 (min length set) and 1 (min length + RANGE)
        :param rest_length: absolute rest length of cable
        :return: control signal
        """
        # Initialize the `done` flag if it is not already set
        if self.done is None:
            self.done = np.array([False])

        # Initialize the cumulative error if it is not already set
        if self.cum_error is None:
            self.cum_error = np.zeros((1), dtype=current_length.dtype)

        # Initialize the control signal `u` as a zero array
        u = np.zeros((1), dtype=current_length.dtype)

        # Clip the cable length range to avoid division by zero
        min_length = self.min_length
        range_ = np.clip(self.RANGE, a_min=1e-5, a_max=999999)

        # Calculate the normalized position of the current cable length
        # NOTE: need to check if the current_length has already been normalized
        # print("DEBUG PID: current_length =", current_length)
        # print("DEBUG PID: target_norm_length =", target_norm_length)
        # print("DEBUG PID: rest_length =", rest_length)
        # print("DEBUG PID: min_length =", min_length)
        # print("DEBUG PID: range_ =", range_)
        position = (current_length - min_length) / range_
        # Perform logic check to verify if the position is within [0, 1] (don't just clip it)
        # if np.any(position < 0.0) or np.any(position > 1.0):
        #     raise ValueError(f"DEBUG PID: Current length {current_length} leads to position {position} out of range [0, 1]")

        # Calculate the target absolute length based on the normalized target length
        target_length = min_length + range_ * target_norm_length

        # Compute the error between the current position and the target normalized length
        error = np.array([position - target_norm_length], dtype=current_length.dtype)

        # Check if the error is within tolerance or if the target is zero and position is below zero
        low_error_cond1 = np.abs(error) < self.tol
        low_error_cond2 = np.abs(current_length - target_length) < 0.1
        low_error_cond3 = np.logical_and(target_norm_length == 0, position < 0)

        # Combine all low-error conditions to determine if the control is "done"
        low_error = np.logical_or(
            np.logical_or(self.done, low_error_cond1),
            np.logical_or(low_error_cond2, low_error_cond3)
        )

        # Update the `done` flag for cables that meet the low-error conditions
        self.done[low_error] = True

        # Calculate the derivative of the error (change in error)
        d_error = np.zeros(error.shape, dtype=error.dtype) \
            if self.last_error is None else error - self.last_error

        # Accumulate the error over time
        self.cum_error += error

        # Store the current error for the next iteration
        self.last_error = error

        # Compute the control signal using the PID formula
        u[~low_error] = (self.k_p * error[~low_error]
                 + self.k_i * self.cum_error[~low_error]
                 + self.k_d * d_error[~low_error])

        # Clip the control signal to the range [-1, 1]
        u = np.clip(u, a_min=-1, a_max=1)

        # Prevent slack in the cable by setting control to 0 if the cable is shorter than its rest length
        slack = np.logical_and(current_length < rest_length, u < 0)
        u[slack] = 0
        print("DEBUG PID: u =", u)

        # Return the control signal and the normalized position
        return u, position

    def reset(self):
        self.last_error = None
        self.cum_error = None
        self.done = None
