import Environment
import numpy as np


print("...... One-Headway-Based Holding Control ......")
num = 20

env = Environment.Env()
b_t = env.b  # boarding time per passenger
a_t = env.a  # alighting time per passenger
q = env.q  # alighting ratio at different stops
c = 0.8  # threshold value
target_H = env.H

headway_cost_list = []

for i in range(num):
    s = env.reset()
    while True:
        # obtain the dwell time
        stop = int(s[1])
        peo_alight = round(s[5] * q[stop - 1])
        alighting_time = peo_alight * a_t
        boarding_time = b_t * s[6]
        dwell_time = max(alighting_time, boarding_time)

        t = s[4] + dwell_time  # current time after boarding/alighting

        dep_t = s[3] + c * target_H  # predicted departure time

        if t < dep_t and stop in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            action = s[3] + target_H - t
        else:
            action = 0
        s_, r, done, _ = env.step(action / env.interval)

        if done:

            break

        s = s_

