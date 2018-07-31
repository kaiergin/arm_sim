import math
import signal
from controller import Controller
import numpy as np
import matplotlib.pyplot as plt
import sys

# constants for simulation
WEIGHT_KG = 1
GRAVITY_N = 9.8
DISTANCE_M = 1
TIMESTEP = 0.01
TRAINING = False
EPOCH_LENGTH = 200 # corresponds to 2 seconds

# maxes and mins for simulation
theta_max = math.pi/2.5
theta_min = -math.pi/2.5
theta_dot_max = 5
theta_dot_min = -5
throttle_max = 20
throttle_min = 0

# calculates ranges
theta_range = theta_max - theta_min
theta_dot_range = theta_dot_max - theta_dot_min
throttle_range = throttle_max - throttle_min

# initial values for simulation
theta = 0
theta_dot = 0

# calculations
force_down = WEIGHT_KG * GRAVITY_N
force_weight = math.cos(theta) * force_down
force_prop = 0
torque = (force_prop - force_weight) * DISTANCE_M

# variables
running = True
done = False
iterations = 0
epochs = 0
q_reward = 0
if len(sys.argv) > 1:
    learning_iterations = int(sys.argv[1])
else:
    learning_iterations = 1000 # default to 10 seconds total

# creates arguments for controller constructor
theta_tuple = (theta_min, theta_max)
theta_dot_tuple = (theta_dot_min, theta_dot_max)
throttle_tuple = (throttle_min, throttle_max)

# lists for final graphs
data = []
graph_q_reward = []
ctrl = Controller(theta_tuple, theta_dot_tuple, throttle_tuple)

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

# used to quickly switch between stages or kill program
def signal_handler(signal, frame):
    global running
    global learning_iterations
    global iterations
    if iterations < learning_iterations:
        iterations = learning_iterations - 1
    else:
        running = False
        print('Stopping')

# the reward function used
def get_reward(theta, theta_dot):
    return sigmoid(theta) * (1 - sigmoid(theta)) +\
        0.5 * sigmoid(theta_dot) * (1 - sigmoid(theta_dot))

print('Stage 1: (Learning on)')
while running:
    # get state of arm
    state = (theta, theta_dot)
    # save state of arm
    prev_state = state
    # get force from controller and put it in boundaries between 0 and 20
    force_prop = ctrl.get_motor_force(state)
    force_prop = max(force_prop, throttle_min)
    force_prop = min(force_prop, throttle_max)
    # counter weight (between 5 and 10?)
    force_weight = math.cos(theta) * force_down
    # calculate changes in state
    torque = (force_prop - force_weight) * DISTANCE_M
    theta_dot += torque * TIMESTEP
    theta_dot = max(theta_dot, theta_dot_min)
    theta_dot = min(theta_dot, theta_dot_max)
    theta += theta_dot * TIMESTEP
    # if changes are out of bounds, set to max/min
    if theta < theta_min:
        theta_dot = 0
        theta = theta_min
        done = True
    if theta > theta_max:
        theta_dot = 0
        theta = theta_max
        done = True
    iterations += 1
    # get state of arm after chosen
    state = (theta, theta_dot)
    # scales theta/theta_dot to calculate reward of state
    theta_scaled = (theta / theta_range) * 3
    theta_dot_scaled = (theta_dot / theta_dot_range) * 3
    # if the state is terminal (out of bounds) sets reward to 0
    if done:
        reward = 0
    else:
        # calculates reward based on square of reward function
        reward = get_reward(theta_scaled, theta_dot_scaled)**2
    # keeps track of total reward for the epoch
    q_reward += reward
    # passes info to controller for training
    ctrl.remember(prev_state, force_prop, reward, state)
    if done:
        done = False
        # if training is on, randomizes theta/theta_dot so the conroller
        # can continue learning
        if TRAINING:
            theta = (2*np.random.rand() - 1) * (math.pi/3)
            theta_dot = 0
    # at end of epoch
    if iterations % EPOCH_LENGTH == 0 and iterations < learning_iterations:
        if TRAINING:
            theta = (2*np.random.rand() - 1) * (math.pi/3)
            theta_dot = 0
        # decays learning rate
        ctrl.decay_epsilon()
        # gives time for controller to learn from epoch
        ctrl.replay()

    # at end of epoch
    if iterations % EPOCH_LENGTH == 0:
        epochs += 1
        # grabs for ctrl+c
        signal.signal(signal.SIGINT, signal_handler)
        print("Epoch:", epochs, "Reward:", q_reward)
        graph_q_reward.append(q_reward)
        q_reward = 0

    if iterations == learning_iterations:
        print('Stage 2: (Learning off)')
        # turns learning of controller off
        ctrl.cut_epsilon()

    data.append(theta)
    # kills program after 1.5 * learning_iterations
    if iterations == int(learning_iterations * 1.5):
        running = False

plt.plot(data)
plt.show()
plt.plot(graph_q_reward)
plt.show()
