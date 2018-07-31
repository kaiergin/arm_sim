import numpy as np

class Controller:
    # each range is a tuple (min, max)
    def __init__(self, theta_range, theta_dot_range, throttle_range):
        pass

    # must return float between 0 and 20
    def get_motor_force(self, state):
        return 0

    # gives the controller info to remember for training
    def remember(self, prev_state, action, reward, current_state):
        pass

    # called at end of epoch, used for training
    def replay(self):
        pass

    # should decay learning rate for controller
    def decay_epsilon(self):
        pass

    # should turn off learning for controller
    def cut_epsilon(self):
        pass

class PidController:
    # each range is a tuple (min, max)
    def __init__(self, theta_range, theta_dot_range, throttle_range):
        self.theta_range = theta_range
        self.theta_dot_range = theta_dot_range
        self.kp = 2.0
        self.ki = 0.0
        self.kd = 0.2
        self.i = 0.0
        self.throttle_range = throttle_range[1] - throttle_range[0]
        self.throttle_min = throttle_range[0]
        self.PidMin = -1.25
        self.PidRange = abs(self.PidMin * 2.0)

    # must return float between 0 and 20
    def get_motor_force(self, state):
        error = state[0]
        de_dt = state[1]
        p = -error
        self.i += -error * 0.01
        d = -de_dt
        pid = self.kp * p + self.ki * self.i + self.kd * d
        thrust = pid * self.throttle_range + self.throttle_min
        print(state, pid, thrust)
        return thrust

    # gives the controller info to remember for training
    def remember(self, prev_state, action, reward, current_state):
        pass

    # called at end of epoch, used for training
    def replay(self):
        pass

    # should decay learning rate for controller
    def decay_epsilon(self):
        pass

    # should turn off learning for controller
    def cut_epsilon(self):
        pass
