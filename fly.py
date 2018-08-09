#!/usr/bin/env python3
# import python libraries
import time
import getopt, sys
import controller as c
import math

# import rcpy library
# This automatically initizalizes the robotics cape
import rcpy
import rcpy.mpu9250 as mpu9250
import rcpy.servo as servo
import rcpy.clock as clock

print("1")

rcpy.set_state(rcpy.RUNNING)
mpu9250.initialize(enable_dmp = True, dmp_sample_rate = 100, enable_fusion = True, enable_magnetometer = True)

srvo = servo.Servo(1)
clck = clock.Clock(srvo, 0.02)
servo.enable()
clck.start()
srvo.set(-1.0)
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

for i in range(3, 0, -1):
    print(i)
    time.sleep(1.0)

data = mpu9250.read()
t0 = time.time()
t00 = time.time()
theta_zero = -data['tb'][0]

try:
    while time.time() < t00 + 5:
         data = mpu9250.read()
         t1 = time.time()
         dt = t1 - t0
         theta = -data['tb'][0] # theta
         d = (theta - theta_zero) / dt # theta_dot

         state = (theta, d)
         throttle = ctrl.get_motor_force(state)

         throttle /= 20
         throttle = -1.0 + throttle*2.0

         throttle = max(throttle, -1.0)
         throttle = min(throttle,  1.0)

         srvo.set(throttle)
         print(throttle, theta, d)
         time.sleep(0.05)
         t0 = t1
         theta_zero = theta
except KeyboardInterrupt:
    srvo.set(-1.0)
srvo.set(-1.0)
print("Done")
