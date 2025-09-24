import numpy as np


class PID:
    def __init__(self,
                 k_p=6.0,
                 k_i=0.01,
                 k_d=0.5,
                 min_length=0.80,
                 RANGE=1.0,
                 tol=0.1,
                 sys_precision=np.float64):
        self.sys_precision = sys_precision
        self.last_error = None
        self.cum_error = None
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.min_length = min_length
        self.RANGE = RANGE
        self.tol = tol
        self.LEFT_RANGE = None
        self.RIGHT_RANGE = None
        self.done = None

    def update_control_by_target_norm_length(self, current_length, target_norm_length, rest_length):
        """
        :param current_length: absolute length of cable
        :param target_norm_length: normalized target length between 0 (min length set) and 1 (min length + RANGE)
        :param rest_length: absolute rest length of cable
        :return: control signal
        """
        if self.done is None:
            self.done = np.array([False])

        if self.cum_error is None:
            self.cum_error = np.zeros((1), dtype=current_length.dtype)

        u = np.zeros((1), dtype=current_length.dtype)

        min_length = self.min_length
        range_ = np.clip(self.RANGE, a_min=1e-5, a_max=999999)

        position = (current_length - min_length) / range_

        # if self.done:
        #     return u, position

        target_length = min_length + range_ * target_norm_length
        error = np.array([position - target_norm_length], dtype=current_length.dtype)

        low_error_cond1 = np.abs(error) < self.tol
        low_error_cond2 = np.abs(current_length - target_length) < 0.1
        low_error_cond3 = np.logical_and(target_norm_length == 0, position < 0)

        low_error = np.logical_or(
            np.logical_or(self.done, low_error_cond1),
            np.logical_or(low_error_cond2, low_error_cond3)
        )

        self.done[low_error] = True

        d_error = np.zeros(error.shape, dtype=error.dtype) \
            if self.last_error is None else error - self.last_error
        self.cum_error += error
        self.last_error = error

        u[~low_error] = (self.k_p * error[~low_error]
                         + self.k_i * self.cum_error[~low_error]
                         + self.k_d * d_error[~low_error])

        u = np.clip(u, a_min=-1, a_max=1)

        slack = np.logical_and(current_length < rest_length, u < 0)
        u[slack] = 0

        return u, position

    def reset(self):
        self.last_error = None
        self.cum_error = None
        self.done = None
