import numpy as np

class Controller:
    def __init__(self, theta_min, theta_max, throttle_min, throttle_max):
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max

    # must return float between 0 and 20
    def get_motor_force(self, state):
        return 0

    # gives the controller info to remember for training
    def remember(self, prev_state, action, reward, current_state):
        return

    # called at end of epoch, used for training
    def replay(self):
        return

    # should decay learning rate for controller
    def decay_epsilon(self):
        return

    # should turn off learning for controller
    def cut_epsilon(self):
        return
