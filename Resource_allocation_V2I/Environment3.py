from __future__ import division
import numpy as np
import random
import math
from beta_allocation import BetaAllocation


n_elements_total = 4
n_elements = 4
t_trans_max =5
RSU_position = [250, 250]


class V2Ichannels:


    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.fc = 2
        self.Decorrelation_distance = 10
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - RSU_position[0])
        d2 = abs(position_A[1] - RSU_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""
    def __init__(self, start_position, start_direction, velocity):
        self.start_position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []

class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_interference_vehicle, BS_width):

        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.BS_width = BS_width
        self.width = width
        self.height = height
        self.interference_vehicle_num = n_interference_vehicle

        self.h_bs = 25
        self.h_ms = 1.5
        self.fc = 2
        self.T_n = 1
        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)  # w
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 11

        self.n_veh = n_veh
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.demand = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2I_pathloss = []

        self.vel_v = []
        self.V2I_channels_abs = []

        self.beta_all = BetaAllocation(self.n_veh)
        self.RSU_f = 6e9


    def renew_position(self):

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.T_n
            change_direction = False
            if self.vehicles[i].direction == 'u':

                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].start_position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].start_position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            self.vehicles[i].start_position = [self.vehicles[i].start_position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].start_position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].start_position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].start_position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                self.vehicles[i].start_position = [self.vehicles[i].start_position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].start_position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].start_position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):

                for j in range(len(self.left_lanes)):  # 向左转
                    if (self.vehicles[i].start_position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].start_position[1] - delta_distance) <= self.left_lanes[j]):  # come to an crossing
                        if (np.random.uniform(0, 1) < 0.3):
                            self.vehicles[i].start_position = [self.vehicles[i].start_position[0] - (
                                        delta_distance - (self.vehicles[i].start_position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].start_position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].start_position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                self.vehicles[i].start_position = [self.vehicles[i].start_position[0] + (
                                        delta_distance + (self.vehicles[i].start_position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].start_position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].start_position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].start_position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            self.vehicles[i].start_position = [self.up_lanes[j], self.vehicles[i].start_position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].start_position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].start_position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].start_position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                self.vehicles[i].start_position = [self.down_lanes[j], self.vehicles[i].start_position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].start_position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].start_position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].start_position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].start_position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            self.vehicles[i].start_position = [self.up_lanes[j], self.vehicles[i].start_position[1] + (
                                    delta_distance - (self.vehicles[i].start_position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].start_position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].start_position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                self.vehicles[i].start_position = [self.down_lanes[j], self.vehicles[i].start_position[1] - (
                                        delta_distance - (self.vehicles[i].start_position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].start_position[0] -= delta_distance

            if (self.vehicles[i].start_position[0] < 0) or (self.vehicles[i].start_position[1] < 0) or (
                    self.vehicles[i].start_position[0] > self.width) or (self.vehicles[i].start_position[1] > self.height):

                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].start_position = [self.vehicles[i].start_position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].start_position = [self.vehicles[i].start_position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].start_position = [self.up_lanes[0], self.vehicles[i].start_position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].start_position = [self.down_lanes[-1], self.vehicles[i].start_position[1]]

            i += 1

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(int(n/4)):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind], 2*self.BS_width-np.random.randint(10, 15)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            vel_v=np.random.randint(10, 15)

            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [self.up_lanes[ind], np.random.randint(10, 15)]
            start_direction = 'u'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [2*self.BS_width-np.random.randint(10, 15), self.left_lanes[ind]]
            start_direction = 'l'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [np.random.randint(10, 15), self.right_lanes[ind]]
            start_direction = 'r'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

        for j in range(int(n % 4)):
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_direction = str
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

    def overall_channel_vel_RSU(self):
        """The combined channel"""
        self.V2I_pathloss = np.zeros((len(self.vehicles)))
        self.vel_v = np.zeros((len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        for i in range(len(self.vehicles)):  # 计算n辆车的路径损失
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].start_position)  #dB为单位
            self.vel_v[i] = self.vehicles[i].velocity
        self.V2I_overall_W = 1/np.abs(1/np.power(10, self.V2I_pathloss / 10))  # W为单位
        '''self.V2I_overall_W = np.power(10, self.V2I_overall_dB / 10)'''
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall_W)+self.V2I_Shadowing  #dB

        return self.V2I_channels_abs

    def overall_channel(self):
        """The combined channel"""
        self.V2I_pathloss = np.zeros((len(self.vehicles)))
        self.vel_v = np.zeros((len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        for i in range(len(self.vehicles)):  # 计算n辆车的路径损失
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].start_position)  #dB为单位
            self.vel_v[i] = self.vehicles[i].velocity
        self.V2I_overall_W = 1/np.abs(1/np.power(10, self.V2I_pathloss / 10))  # W为单位
        '''self.V2I_overall_W = np.power(10, self.V2I_overall_dB / 10)'''
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall_W)+self.V2I_Shadowing  #dB

        return self.V2I_channels_abs

    def true_calculate_num(self, action_pf):
        comp_n_list_true, comp_n_list = self.beta_all.true_calculate_times(action_pf)
        return comp_n_list_true, comp_n_list

    def calculate_num_RSU(self):
        calculate_num_RSU = self.beta_all.calculate_times_RSU(self.RSU_f)
        return calculate_num_RSU

    def update_buffer(self, comp_n_list):
        for i in range(self.n_veh):
            self.ReplayB_v[i] += comp_n_list[i]

    def trans_energy_RSU(self, action_pf, h_i_dB):
        p_selection = action_pf[:, 0].reshape(len(self.vehicles), 1)

        V2I_Signals = np.zeros(self.n_veh)

        V2I_Interference = 0

        for i in range(len(self.vehicles)):

            self.V2I_Interference = V2I_Interference + self.sig2  # 单位：W

            V2I_Signals[i] = 10 ** ((p_selection[i] - self.V2I_pathloss_with_fastfading[i]+ self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)  #单位：W
        trans_energy_RSU = self.beta_all.trans_energy_RSU(action_pf, h_i_dB, self.V2I_Interference)

        return trans_energy_RSU

    def renew_channel_fastfading(self):

        self.V2I_pathloss_with_fastfading = 20 * np.log10(
            np.abs(np.random.normal(0, 1, self.V2I_channels_abs.shape) + 1j * np.random.normal(0, 1, self.V2I_channels_abs.shape)) / math.sqrt(2))

        self.V2I_pathloss_with_fastfading = self.V2I_channels_abs - self.V2I_pathloss_with_fastfading

    def RSU_reward1(self, action_pf, comp_n_list_true, trans_energy_RSU, offload_num):

        E_total = np.zeros(self.n_veh)
        RSU_energy = self.beta_all.energy_RSU(self.RSU_f)
        E_single_energy_vel = self.beta_all.single_comp_energy_vel(action_pf)

        cf = 0
        overload = []
        load_rate_0 = []
        ReplayB_v_copy = list(self.ReplayB_v)

        for i in range(self.n_veh):
            uu=1
            if self.ReplayB_v[i]>offload_num[i]:
                if offload_num[i] ==0:
                    uu=0
                load_rate_0.append(offload_num[i])

                self.ReplayB_v[i] -= offload_num[i]

                if self.ReplayB_v[i]>comp_n_list_true[i]:
                    E_total[i] = comp_n_list_true[i]*E_single_energy_vel[i]+ uu*trans_energy_RSU[i] + offload_num[i]* RSU_energy
                    self.ReplayB_v[i] -= comp_n_list_true[i]

                else:
                    E_total[i] = offload_num[i]* RSU_energy + uu*trans_energy_RSU[i] + self.ReplayB_v[i]*E_single_energy_vel[i]
                    self.ReplayB_v[i] = 0
            else:
                cf = 0.001

                load_rate_0.append(self.ReplayB_v[i])

                overload.append(offload_num[i] - self.ReplayB_v[i])
                offload_num[i] = self.ReplayB_v[i]
                E_total[i] = offload_num[i] * RSU_energy + uu*trans_energy_RSU[i]
                self.ReplayB_v[i] = 0

        array1 = np.array(load_rate_0)
        array2 = np.array(ReplayB_v_copy)
        rate_0_temp = []
        for i in range(len(array1)):
            if array2[i] == 0:
                continue
            rate_0_temp.append(array1[i] / array2[i])

        rate_0 = [x for x in rate_0_temp if x != 0]

        reward_tot = 10*sum(E_total) + cf * sum(overload) + 0.01 * sum(self.ReplayB_v)

        return E_total, reward_tot, sum(overload), rate_0


    def get_state(self):
        """
        Get channel information from the environment
        """
        bb = sum(self.V2I_channels_abs)
        V2I_abs= self.V2I_channels_abs/bb

        vel_v = self.vel_v /20

        return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(vel_v, -1)))
        # size=4+1+1

    def Compute_Performance_Reward_Train(self, action_pf, h_i_dB, vel_v, lambda_1, lambda_2):

        p_selection = action_pf[:, 0].reshape(len(self.vehicles), 1)
        V2I_Signals = np.zeros(self.n_veh)

        V2I_Interference = self.sig2 # 单位：W

        for i in range(len(self.vehicles)):
            V2I_Signals[i] = 10 ** ((p_selection[i] - self.V2I_channels_abs[i]+ self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        E_total, tran_success, c_total, reward, comp_n_list = self.beta_all.beta_allocation(action_pf, h_i_dB, vel_v, lambda_1, lambda_2, V2I_Interference)

        return E_total, tran_success, c_total, reward, comp_n_list


    def act_for_training(self, action_pf, h_i_dB, vel_v, lambda_1, lambda_2):
        E_total, tran_success, c_total, reward, comp_n_list = self.Compute_Performance_Reward_Train(action_pf, h_i_dB, vel_v, lambda_1, lambda_2)

        return E_total, tran_success, c_total, reward, comp_n_list

    def new_random_game(self, n_veh=0):

        self.vehicles = []
        self.vehicles_interference = []
        if n_veh > 0:
            self.n_veh = n_veh
        self.add_new_vehicles_by_number(self.n_veh)
        self.overall_channel()
        self.renew_channel_fastfading()
        self.ReplayB_v = np.zeros(self.n_veh)
