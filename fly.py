#!/usr/bin/env python3
# import python libraries
import time
import getopt, sys
import conroller as c

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
ctrl = c.QTblController(1,1,1)
ctrl.load_table()

for i in range(3, 0, -1):
    print(i)
    time.sleep(1.0)

data = mpu9250.read()
t0 = time.time()
theta_zero = data['tb'][0]

try:
    while True:
         data = mpu9250.read()
         t1 = time.time()
         dt = t1 - t0
         theta = data['tb'][0] # theta
         d = (theta - theta_zero) / dt # theta_dot

         state = (theta, d)
         throttle = ctrl.get_motor_force(state)

         throttle /= 20
         throttle = -1.0 + throttle*2.0

         throttle = max(throttle, -1.0)
         throttle = min(throttle,  1.0)

         #srvo.set(throttle)
         print(throttle)
         time.sleep(0.05)
         t0 = t1
         theta_zero = theta

except KeyboardInterrupt:
    while throttle > -0.7:
        print('shutdown:', throttle)
        throttle = throttle - 0.05
        srvo.set(throttle)
        time.sleep(100.0)
    srvo.set(-1.0)
