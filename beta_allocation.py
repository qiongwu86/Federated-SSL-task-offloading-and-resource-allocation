import math
import numpy as np
import random


class BetaAllocation:
    def __init__(self, n_veh):
        # 全局变量
        self.n_veh = n_veh
        self.Z = 11.2
        self.B = 2000000
        self.h_n_dB = 20 * math.log10((4 * math.pi * 200 * 915e6) / 3e8)
        self.I_n = 0
        self.N_0 = 10 ** (-114 / 10)
        self.T_n = 1
        self.data_t = 0.02
        self.k = 1e-27
        self.r_n = 1600
        self.D_n = 1500
        self.q_tao = 0.2
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 11
        self.m = 0.023

    def true_calculate_times(self, p_f):
        list_num_true = []
        list_num = []
        for i in range(self.n_veh):
            random_value = random.uniform(0.05,1)
            T_true = random_value * self.T_n
            T_comp = self.D_n * self.r_n /p_f[i][1]

            num =round((self.T_n-self.data_t)/T_comp)
            num_true = round((T_true-self.data_t)/T_comp)

            list_num_true.append(num_true)
            list_num.append(num)
        return list_num_true, list_num

    def calculate_times_RSU(self, RSU_f):
        calculate_times_RSU = round((self.T_n-self.data_t)/(self.D_n * self.r_n /RSU_f))
        return calculate_times_RSU

    def energy_RSU(self, RSU_f):
        E = self.k * self.D_n * self.r_n * RSU_f**2
        return E

    def trans_energy_RSU(self, p_f, h_i_dB, V2I_Interference):
        V2I_Signals_dB = np.zeros(self.n_veh)
        V2I_Signals_W = np.zeros(self.n_veh)
        trans_energy_RSU = np.zeros(self.n_veh)
        count = 0
        for p_f_1 in p_f:

            V2I_Signals_dB[count] = p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
            V2I_Signals_W[count] = 10 ** (
                        (p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

            trans_energy_RSU[count] = p_f_1[0]*(self.Z+self.D_n) / self.B/len(trans_energy_RSU) / math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)
            count += 1
        return trans_energy_RSU

    def single_comp_energy_vel(self,pf):
        E_list = []
        for i in range(self.n_veh):
            E = self.k * self.D_n * self.r_n * pf[i][1] ** 2
            E_list.append(E)
        return E_list