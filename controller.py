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
