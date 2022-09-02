import pandas as pd
import matplotlib.pyplot as plt
import os

def save_data(data_set, data_path):  # num_row=num_variables, num_row=num_steps
    df = pd.DataFrame(data=data_set)
    # print(df)
    df.to_csv(data_path, header=False, index=False)


def read_data(data_path):
    df = pd.read_csv(data_path, header=None)
    data = df.values
    return data


def plot_data(data, state_dim, action_dim, n_step):
    state_label = ["T_w", "T_in", "T_g", "T_p", "T_1", "T_2", "T_3", "E"]
    action_label = ["P_ch", "P_dis", "P_buy", "P_sell", "P_hp", "X_v", "P_pv", "P_app", "Price"]
    state = data[:state_dim, :]   # e.g. 8*2880
    action = data[state_dim:state_dim + action_dim, :]
    lspo = data[state_dim + action_dim:state_dim + action_dim + 1, :]
    ltem = data[state_dim + action_dim + 1:state_dim + action_dim + 2, :]
    reward = data[state_dim + action_dim + 2:state_dim + action_dim + 3, :]
    rollout_return = data[state_dim + action_dim + 3:, :]

    plt.figure(1)
    # plt.ion()
    for i in range(state_dim):
        plt.subplot(3, 3, i + 1)
        plt.plot(range(n_step), state[i, :], c='b')
        plt.ylabel(state_label[i])
    # plt.pause(0.1)

    plt.figure(2)
    # plt.ion()
    for i in range(action_dim):
        plt.subplot(3, 3, i + 1)
        plt.plot(range(n_step), action[i, :], c='b')
        plt.ylabel(action_label[i])
    # plt.pause(0.1)

    plt.figure(3)
    # plt.ion()
    plt.subplot(2, 2, 1)
    plt.scatter(range(n_step), lspo, c='b')
    plt.ylabel("l_spo")
    plt.subplot(2, 2, 2)
    plt.scatter(range(n_step), ltem, c='b')
    plt.ylabel("l_tem")
    plt.subplot(2, 2, 3)
    plt.scatter(range(n_step), reward, c='b')
    plt.ylabel("reward")
    plt.subplot(2, 2, 4)
    plt.scatter(range(n_step), rollout_return, c='b')
    plt.ylabel("rollout_return")
    # plt.pause(0.1)

    plt.show()

#################
if __name__ == "__main__":
    ### read data and plot
    SafeRL_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    # save_path = os.path.join(SafeRL_path, 'Project/Results/SmartHome/result_mpc_no_noise.csv')
    save_path = os.path.join(SafeRL_path, 'Project/Results/SmartHome/result_mpc_noise.csv')
    # save_path = os.path.join(SafeRL_path, 'Project/Results/SmartHome/result_mpc_test.csv')
    data = read_data(save_path)
    plot_data(data, state_dim=8, action_dim=6+3, n_step=96)