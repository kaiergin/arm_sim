import math
import time
import signal
from controller import Controller
import numpy as np
import matplotlib.pyplot as plt

# constants for simulation
WEIGHT_KG = 1
GRAVITY_N = 9.8
DISTANCE_M = 1
TIMESTEP = 0.01
TRAINING = False

# maxes and mins for simulation
theta_max = math.pi/2.5
theta_min = -math.pi/2.5
theta_dot_max = 2
theta_dot_min = -2
min_throttle = 0
max_throttle = 20
action_space = (max_throttle - min_throttle) * 2

# initial values for simulation
theta = (2*np.random.rand() - 1) * (math.pi/3)
theta_dot = 0

# Calculations
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
learning_iterations = 1000000

# calculates ranges and buckets (not used anymore)
theta_range = (theta_max - theta_min)
theta_dot_range = (theta_dot_max - theta_dot_min)
theta_step = theta_range / 40
theta_dot_step = theta_range / 40
theta_bucket = int(theta_range // theta_step)
theta_dot_bucket = int(theta_dot_range // theta_dot_step)

# lists for final graphs
data = []
graph_q_reward = []
ctrl = Controller(theta_min, theta_max, min_throttle, max_throttle)

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
    # get force from controller
    force_prop = ctrl.get_motor_force(state)
    force_prop = max(force_prop, min_throttle)
    force_prop = min(force_prop, max_throttle)
    # counter weight (between 5 and 10?)
    force_weight = math.cos(theta) * force_down
    # calculate changes in state
    torque = (force_prop - force_weight) * DISTANCE_M
    theta_dot += torque * TIMESTEP
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
    theta_dot_scaled = min(theta_dot, theta_dot_max)
    theta_dot_scaled = max(theta_dot, theta_dot_min)
    theta_dot_scaled = (theta_dot_scaled / theta_dot_range) * 3
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
    if iterations % 200 == 0 and iterations < learning_iterations:
        if TRAINING:
            theta = (2*np.random.rand() - 1) * (math.pi/3)
            theta_dot = 0
        # decays learning rate
        ctrl.decay_epsilon()
        # gives time for controller to learn from epoch
        ctrl.replay()

    # at end of epoch
    if iterations % 200 == 0:
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
