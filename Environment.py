import random

import pandas as pd
import numpy as np


class Env(object):

    def __init__(self, travel_time_std):
        # np.random.seed(20201229)
        simulation_time = 4
        self.bus_fleet = 6
        self.number_of_buses = self.bus_fleet * simulation_time  # set number of bus trips
        self.number_of_stops = 11  # set number of stops
        self.lamb = np.array([0.5, 1.4, 4.5, 4, 1.8, 1.45, 1.05, 0.75, 0.45, 0.2, 0.5]) / 60  # pax/s
        self.q = np.array([1, 0, 0.25, 0.25, 0.5, 0.5, 0.1, 0.75, 0.2, 0.1, 1])
        self.b = 3.  # s/pax
        self.a = 1.8  # s/pax
        self.H = 6 * 60  # the planned headway
        self.travel_time_mean = 180.
        self.travel_time_std = travel_time_std
        self.demand_std = 0.1
        self.interval = 30
        self.done = None
        self.action_space = np.arange(0, 91, self.interval)  # 4 actions: 0, 30, 45, 60
        self.df = pd.DataFrame(index=range(1, self.number_of_buses + 1))  # create dataframe
        self.ini_state = None
        self.control_list = pd.DataFrame(index=range(2, self.number_of_buses + 1),
                                         columns=range(1, self.number_of_stops + 1))
        self.waiting_people = pd.DataFrame(index=range(1, self.number_of_buses + 1),
                                           columns=range(1, self.number_of_stops + 1))
        self.onboard_people = pd.DataFrame(index=range(1, self.number_of_buses + 1),
                                           columns=range(1, self.number_of_stops + 1))

    def reset(self):
        self.done = False
        arr_vars = []  # create arrival time variables when arriving at stops
        dep_vars = []

        for i in range(1, self.number_of_stops + 1):
            varname1 = "arr_" + str(i)
            self.df[varname1] = 0.
            arr_vars.append(varname1)

            varname2 = "dep_" + str(i)
            self.df[varname2] = 0.
            dep_vars.append(varname2)

        # create the departure times at stop 1
        self.df['arr_1'] = np.arange(self.number_of_buses) * self.H

        # create the track of the first bus trip (bus trip 1)
        for arr, dep in zip(arr_vars, dep_vars):
            # the index of stop
            a = int(arr.split("_")[1])
            if a == 1:
                # the arrival time at stop 1
                self.df.loc[1, arr] = 0
                self.onboard_people.loc[1, a] = 0
            else:
                # # create travel time
                # travel_time = np.random.lognormal(5.2, 0.1, 10)
                travel_time = np.random.normal(self.travel_time_mean, self.travel_time_std * self.travel_time_mean)

                # obtain the arrival time at other stops
                self.df.loc[1, arr] = self.df.loc[1, 'dep_' + str(a - 1)] + travel_time

            # the number of people alighting at stop a
            peo_alight = round(self.onboard_people.loc[1, a] * self.q[a - 1])

            # estimate the number of people waiting at stop a
            peo_wait = np.sum(np.random.poisson(self.lamb[a - 1], self.H))
            self.waiting_people.loc[1, a] = peo_wait

            # the number of people after finishing alighting and boarding
            self.onboard_people.loc[1, a + 1] = self.onboard_people.loc[1, a] - peo_alight + peo_wait

            # create dwell time
            dwell_time = max(peo_alight * self.a, peo_wait * self.b)

            # get the departure time
            self.df.loc[1, dep] = self.df.loc[1, arr] + dwell_time

        # # initial state
        # arrival_time of bus 2 at stop 1
        arrival_time_2_1 = self.df.loc[2, 'arr_1']
        headway = arrival_time_2_1 - self.df.loc[1, 'arr_1']
        peo_2_1 = np.sum(np.random.poisson(self.lamb[1 - 1], headway))

        '''
        state method 1:
        [bus trip, stop,
         arrive time of preceding bus trip, departure time of preceding bus trip, arrive time of the examined trip,
          people onboard, people waiting to board when bus arriving at the stop]
        '''
        self.ini_state = [2, 1, self.df.loc[1, 'arr_1'], self.df.loc[1, 'dep_1'], self.df.loc[2, 'arr_1'], 0, peo_2_1]

        '''
        state method 2:
        [bus trip, stop, arrive time of preceding bus trip, arrive time of the examined trip, people]
        '''

        return np.array(self.ini_state)

    def step(self, act):

        state = self.ini_state
        # print(state)
        b, s, a, d, a_, p_on, p_wb = int(state[0]), int(state[1]), state[2], state[3], state[4], int(state[5]), int(state[6])

        # # headway of last stop
        # pre_headway = a_ - a

        # get the waiting people at different stops
        self.waiting_people.loc[b, s] = p_wb
        # get the onboard people at different stops
        self.onboard_people.loc[b, s] = p_on

        # 1. get the boarding time
        boarding_time = p_wb * self.b
        # 2. get the alighting time
        # the number of passengers alighting
        peo_alight = round(p_on * self.q[s - 1])
        alighting_time = peo_alight * self.a

        # get the dwell time of bus n at stop s
        dwell_time = max(boarding_time, alighting_time)

        holding_time = act * self.interval

        # get the holding time at different stops
        self.control_list.loc[b, s] = holding_time

        # get the departure time of bus n at stop s
        dep_time = a_ + dwell_time + holding_time
        self.df.loc[b, 'dep_' + str(s)] = dep_time

        # update the people onboard when leaving stop s (that is the same number of people when bus arriving at
        # next stop s+1)
        p_on = p_on - peo_alight + p_wb

        # move to the next stop
        # judge whether reach the end of stop

        # judge whether the last bus has reached at the last stop
        if b == self.number_of_buses and s == self.number_of_stops:
            self.ini_state = 'terminal'
            reward = 0
            self.done = True

        else:

            if b < self.bus_fleet and s == self.number_of_stops:
                b += 1
                s = 1
                p_on = 0
                # obtain the arrival time of the bus (1-6) at stop 1
                arr_time = self.df.loc[b, 'arr_' + str(s)]

            elif b >= self.bus_fleet and s == self.number_of_stops:
                b += 1
                s = 1
                p_on = self.onboard_people.loc[b - self.bus_fleet, self.number_of_stops]
                # obtain the arrival time of the bus (>6) at stop 1
                arr_time = self.df.loc[b - self.bus_fleet, 'arr_' + str(self.number_of_stops)]

            else:
                b = b
                s += 1

                # generate travel time to the next stop
                # travel_time = np.random.lognormal(5.2, 0.1, 10)
                travel_time = np.random.normal(self.travel_time_mean, self.travel_time_std * self.travel_time_mean)

                # obtain the arrival time of the next state
                arr_time = self.df.loc[b, 'dep_' + str(s - 1)] + travel_time

            # update the arrival and departure time of ahead bus at this stop
            a = self.df.loc[b - 1, 'arr_' + str(s)]
            d = self.df.loc[b - 1, 'dep_' + str(s)]

            # no overtaking and get the arrival time when bus arrive at stop
            if arr_time < d:
                self.df.loc[b, 'arr_' + str(s)] = d
            else:
                self.df.loc[b, 'arr_' + str(s)] = arr_time
            a_ = self.df.loc[b, 'arr_' + str(s)]

            # arrival headway of this stop
            headway = a_ - a

            # calculate the headway variation reward and holding reward
            reward = - abs(headway - self.H) - holding_time / 3
            # reward = abs(pre_headway - self.H) - abs(headway - self.H)

            # get the number of people waiting at this stop
            time = int(a_ - a)
            p_wb = np.sum(np.random.poisson(self.lamb[s - 1], time)).item()
            # p_wb = random.gauss(p_wb, p_wb*self.demand_std)

            # update the state
            self.ini_state = [b, s, a, d, a_, p_on, p_wb]

        return np.array(self.ini_state), reward, self.done, {}


if __name__ == "__main__":

    def cal_headway_variation_cost(data):
        headway = data - data.shift(1)
        headway = (headway - 360) ** 2
        arr_vars = []
        for i in range(2, 11):
            var_name = "arr_" + str(i)
            arr_vars.append(var_name)

        arr_headway = headway.drop(index=1)
        arr_headway = arr_headway[arr_vars]
        cost = np.sqrt(arr_headway.mean().mean())
        return cost

    num = 20

    # STD = np.linspace(0, 0.2, 11)
    # headway_variation_df = pd.DataFrame(index=range(20), columns=STD)
    STD = [0.1]

    for std in STD:
        print("std: %.2f" % std)
        headway_cost_list = []

        for i in range(num):
            # print("------NoHolding episode: %d------" % i)

            env = Env(std)
            s = env.reset()
            ep_r = 0

            while True:
                stop = int(s[1])
                action = 0
                s_, r, done, _ = env.step(action)

                if done:
                    cost_headway = cal_headway_variation_cost(env.df)
                    headway_cost_list.append(cost_headway)

                    break

                if stop in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    ep_r += r

                s = s_

            env.df.to_csv("noholding_traj_v0" + str(i) + ".csv")
            # env.onboard_people.to_csv("noholding_onboard_people_v" + str(i) + ".csv")

        # data_reward = pd.DataFrame(reward)
        # print("reward: %.2f" % np.mean(reward))
        # data_reward.to_csv("noholding_0.2_reward.csv")

        # headway_cost = pd.DataFrame(cost_headway_list)
        # print("headway_cost: %.2f" % np.mean(cost_headway_list))
        # headway_cost.to_csv("noholding_headway_cost.csv")

        # headway_variation_df.loc[:, std] = headway_cost_list

    # headway_variation_df.to_csv('headway_variation_nh.csv')

    # travel_cost = pd.DataFrame(cost_travel_time_list)
    # print("travel_time_cost: %.2f" % np.mean(cost_travel_time_list))
    # travel_cost.to_csv("noholding_0.2_travel_time_cost.csv")

    # plt.figure(figsize=(15, 12))
    #
    # plt.subplot(221)
    # plt.grid()
    # plt.title("noholding_0.2_reward")
    # plt.plot(reward)
    #
    # plt.subplot(223)
    # plt.grid()
    # plt.title("noholding_0.2_headway cost")
    # plt.plot(range(1, num + 1), cost_headway_list)
    #
    # plt.subplot(222)
    # plt.grid()
    # plt.title("noholding_0.2_travel time cost")
    # plt.plot(range(1, num + 1), cost_travel_time_list)
    #
    # plt.suptitle("noholding_0.2")
    #
    # plt.show()

