import controller as c
import math

# for testing the saved q table

theta_max = math.pi/2.5
theta_min = -math.pi/2.5
theta_dot_max = 5
theta_dot_min = -5
throttle_max = 20
throttle_min = 0
theta_tuple = (theta_min, theta_max)
theta_dot_tuple = (theta_dot_min, theta_dot_max)
throttle_tuple = (throttle_min, throttle_max)
ctrl = c.QTblController(theta_tuple, theta_dot_tuple, throttle_tuple)
ctrl.load_table()

values = [-.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5]

for x in values:
    for y in values:
        state = (x,y)
        print("theta:", x, "theta_dot:", y, "force:", ctrl.get_motor_force(state))
