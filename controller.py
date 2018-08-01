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

class QTblController:
    # each range is a tuple (min, max)
    def __init__(self, theta_range, theta_dot_range, throttle_range):
        self.theta_min, self.theta_max = theta_range
        self.theta_range = theta_range[1] - theta_range[0]
        self.theta_dot_min, self.theta_dot_max = theta_dot_range
        self.theta_dot_range = theta_dot_range[1] - theta_dot_range[0]
        self.extra_action = 2
        self.action_min, self.action_max = throttle_range
        self.action_space = int((throttle_range[1] - throttle_range[0]) * self.extra_action)
        self.theta_step = 0.1
        self.theta_dot_step = 0.1
        self.theta_buckets = int(self.get_theta_buckets(self.theta_range))
        self.theta_dot_buckets = int(self.get_theta_dot_buckets(self.theta_dot_range))
        self.q_table = np.zeros((self.theta_buckets, self.theta_dot_buckets, self.action_space))
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.alpha = 0.01
        self.epsilon_min = 0.01
        self.gamma = 0.95

    # based on the range of theta and the size of theta_step, gives number of buckets
    def get_theta_buckets(self, range):
        num_buckets = range // self.theta_step
        return num_buckets + 1

    def get_theta_dot_buckets(self, range):
        num_buckets = range // self.theta_dot_step
        return num_buckets + 1

    def get_output(self, action):
        return action / self.extra_action

    # get bucket that theta falls into
    def get_t_bucket(self, theta):
        theta -= self.theta_min
        return int(theta // self.theta_step)

    def get_td_bucket(self, theta_dot):
        theta_dot -= self.theta_dot_min
        return int(theta_dot // self.theta_dot_step)

    def get_bucket(self, state):
        state = (self.get_t_bucket(state[0]), self.get_td_bucket(state[1]))
        return state

    # must return float between 0 and 20
    def get_motor_force(self, current_state):
        if np.random.rand() < self.epsilon:
            return self.get_output(np.random.randint(self.action_space))
        else:
            current_state = self.get_bucket(current_state)
            return self.get_output(np.argmax(self.q_table[current_state]))

    # gives the controller info to remember for training
    def remember(self, prev_state, action, reward, current_state):
        prev_state_bucket = self.get_bucket(prev_state)
        current_state_bucket = self.get_bucket(current_state)
        action = int(action * self.extra_action)
        '''
        self.q_table[prev_state_bucket][action] +=                     \
                self.alpha * (reward + self.gamma *                    \
                np.amax(self.q_table[current_state_bucket]) -          \
                self.q_table[prev_state_bucket][action])
        '''
        hold = reward if reward == 0 else reward + self.gamma * np.amax(self.q_table[current_state_bucket])
        self.q_table[prev_state_bucket][action] = hold

    # called at end of epoch, used for training
    def replay(self):
        pass

    # should decay learning rate for controller
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    # should turn off learning for controller
    def cut_epsilon(self):
        self.epsilon = 0.0

    def save_table(self):
        save_file = open('q_table','wb')
        np.save(save_file, self.q_table)

    def load_table(self, table):
        load_file = open(table, 'rb')
        self.q_table = np.load(load_file)
        self.epsilon = 0.0
        self.epsilon_min = 0.0
