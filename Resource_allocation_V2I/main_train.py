
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import Environment3
from RL_train5 import SAC_Trainer
from RL_train5 import ReplayBuffer


BS_width = 1000/2
up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
print(up_lanes)
print(down_lanes)
print(left_lanes)
print(right_lanes)


width = 1000
height = 1000

B = 2e6
sig1=-144  # 噪声功率：dB
sig2=10**(sig1/10)
q_tao = 0.5

BS_position = [0, 0]
max_power = 200
min_power = 5
max_f = 4e8
min_f = 5e7
m = 0.023  # dB


V2I_min = 3.16
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

batch_size = 64
memory_size = 1000000


n_step_per_episode = 100
n_episode_test = 3000  # test episodes大循环
n_interference_vehicle = 0
n_veh =5
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
n_input = 2 * n_veh  # 8
n_output = 3 * n_veh

n_input_RSU = 2 * n_veh  # 8
n_output_RSU = 2 * n_veh
# --------------------------------------------------------------

replay_buffer_size = 1e6
replay_buffer_size_RSU = 1e6

hidden_dim = 512
action_range = 1.0
AUTO_ENTROPY = True
DETERMINISTIC = False

#---------------------------------------------------------------
update_timestep = 100  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor
lr = 0.01  # parameters for Adam optimizer
betas = (0.9, 0.999)

#---------------------------------------------------------------
fc1_dims = 512
fc2_dims = 512
fc3_dims = 512
fc4_dims = 512
alpha = 0.0001
beta = 0.001
tau = 0.05
pp = 0.005

def SAC_train(ii):
    print("\nRestoring the SAC model...")
    # --------------model--------------
    model_path = 'model\SAC_model'
    replay_buffer = ReplayBuffer(replay_buffer_size)
    RL_SAC = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)

    Sum_E_total_list = []
    Sum_reward_list = []
    Sum_calculate_list = []
    Sum_overload_list = []

    Sum_eta1_list = []
    Sum_load_rate_0_episode_list = []

    Vehicle_positions_x = [[] for _ in range(n_veh)]
    Vehicle_positions_y = [[] for _ in range(n_veh)]

    for i_episode in range(n_episode_test):  #
        if i_episode ==50:
            print('A')
        print('------ SAC/Episode', i_episode, '------')

        env.new_random_game()

        for i in range(n_veh):
            Vehicle_positions_x[i].append(env.vehicles[i].start_position[0])
            Vehicle_positions_y[i].append(env.vehicles[i].start_position[1])

        state_old_all = []

        state = env.get_state()
        state_old_all.append(state)

        Sum_E_total_per_episode = []
        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_overload_per_episode = []

        Sum_load_rate_0_episode = []

        eta1 = []
        for i_step in range(n_step_per_episode):
            env.renew_position()

            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, 3], dtype=np.float64)

            action = RL_SAC.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)

            for i in range(n_veh):
                action_all_training[i, 0] = ((action[0 + i * 2] + 1) / 2) * (max_power-min_power)+min_power
                action_all_training[i, 1] = ((action[1 + i * 2] + 1) / 2) * (max_f-min_f)+min_f
                action_all_training[i, 2] = (action[2 + i * 2] + 1) / 2

            action_pf = action_all_training.copy()

            comp_n_list_true, comp_n_list = env.true_calculate_num(action_pf)

            comp_n_list_RSU = env.calculate_num_RSU()

            env.update_buffer(comp_n_list)

            offload_num = []
            for i in range(n_veh):
                offload_num_i = int(action_pf[i, 2]*comp_n_list_RSU)
                offload_num.append(offload_num_i)

            h_i_dB= env.overall_channel()

            trans_energy_RSU = env.trans_energy_RSU(action_pf, h_i_dB)

            E_total, reward_tot, overload, load_rate_0 = env.RSU_reward1(action_pf, comp_n_list_true, trans_energy_RSU, offload_num)

            eta1.append(overload/sum(comp_n_list))

            if load_rate_0==[]:
                load_rate_0 = np.ones(n_veh)

            reward = -1 * reward_tot
            E_total = 1 * E_total

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))

            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)

            state_new = env.get_state()

            state_new_all.append((state_new))

            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           reward, np.asarray(state_new_all).flatten(), 0)

            if len(replay_buffer) > 256:  # batch_size
                for i in range(1):
                    _ = RL_SAC.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*n_output)

            state_old_all = state_new_all

        Sum_E_total_list.append((np.mean(Sum_E_total_per_episode)))
        Sum_reward_list.append((np.mean(Sum_reward_per_episode)))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))

        Sum_eta1_list.append((np.mean(eta1)))
        Sum_load_rate_0_episode_list.append((np.mean(Sum_load_rate_0_episode)))

        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode),6))

    RL_SAC.save_model(model_path)

    return Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list



if __name__ == "__main__":


    for i in range(1):
        name = 'SAC'
        env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_interference_vehicle, BS_width)

        env.new_random_game()
        E_total_list, reward_list, calculate_list, overload_list, buffer_list, eta1_list, load_rate_0_episode_list = SAC_train(i)

        save_results(name, i, E_total_list, reward_list, calculate_list, overload_list, buffer_list, eta1_list, load_rate_0_episode_list)
