import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from matplotlib.pyplot import Polygon


def plot_data(data1, data2, state_dim, action_dim, n_step):
    state_label = [r"$T_{\rm{w}}$", r"$T_{\rm{in}}$", r"$T_{\rm{g}}$", r"$T_{\rm{p}}$", r"$T_1$", r"$T_2$",
                   r"$T_3$", r"$E$"]
    action_label = [r"$P_{\rm{ch}}$", r"$P_{\rm{dis}}$", r"$P_{\rm{buy}}$", r"$P_{\rm{sell}}$",
                    r"$P_{\rm{hp}}$", r"$X_{\rm{v}}$"]
    unc_label = [r"$P_{\rm{pv}}$", r"$P_{\rm{app}}$", r"$T_{\rm{out}}$", "Price"]

    state1 = data1[:state_dim, :]  # e.g. 8*2880
    state2 = data2[:state_dim, :]  # e.g. 8*2880
    action1 = data1[state_dim:state_dim + action_dim, :]
    action2 = data2[state_dim:state_dim + action_dim, :]
    rollout_return1 = data1[state_dim + action_dim: state_dim + action_dim + 1, :].squeeze()
    rollout_return2 = data2[state_dim + action_dim: state_dim + action_dim + 1, :].squeeze()
    rollout_return_spo1 = data1[state_dim + action_dim + 1: state_dim + action_dim + 2, :].squeeze()
    rollout_return_spo2 = data2[state_dim + action_dim + 1: state_dim + action_dim + 2, :].squeeze()
    rollout_return_temp1 = data1[state_dim + action_dim + 2: state_dim + action_dim + 3, :].squeeze()
    rollout_return_temp2 = data2[state_dim + action_dim + 2: state_dim + action_dim + 3, :].squeeze()
    uncs1 = data1[state_dim + action_dim + 3:, :]  # 4*2880
    uncs2 = data2[state_dim + action_dim + 3:, :]

    time = np.arange(n_step)/4

    plt.figure(1)
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        if i == 1:
            plt.plot(time, state1[i, :], label=r'$\theta_0$')
            plt.plot(time, state2[i, :], label=r'$\theta^{\star}$')
            # plt.plot(time, [23] * n_step, linestyle='dashed', color='r', linewidth=0.5)
            # plt.plot(time, [27] * n_step, linestyle='dashed', color='r', linewidth=0.5)
            plt.fill_between(x=time, y1=[23] * n_step, y2=[27] * n_step, color='gray',
                             alpha=0.2)
        else:
            plt.plot(time, state1[i, :], label=r'$\theta_0$')
            plt.plot(time, state2[i, :], label=r'$\theta^{\star}$')
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel(state_label[i])
        plt.xlim((0, 23))


    fig, ax = plt.subplots(3, 2)
    ai = 0
    for i in range(3):
        for j in range(2):
            if not ai == 5:
                ax[i, j].plot(time, action1[ai, :], label=r'$\theta_0$')
                ax[i, j].plot(time, action2[ai, :], label=r'$\theta^{\star}$')
                ax_price = ax[i, j].twinx()
                # ax_price.plot(time, uncs1[3, :], label='price', color='g')
                ax_price.fill_between(x=time, y1=uncs1[3, :], color='gray', alpha=0.1)
                ax[i, j].legend()
                ax[i, j].set_xlabel('Hour')
                ax[i, j].set_ylabel(action_label[ai])
                ax[i, j].set_xlim((0, 23))
                ax_price.set_ylim((3.5, 13))
                ax_price.set_yticks([])
                ai += 1
            else:
                ax[i, j].plot(time, action1[ai, :], label=r'$\theta_0$')
                ax[i, j].plot(time, action2[ai, :], label=r'$\theta^{\star}$')
                ax_price = ax[i, j].twinx()
                # ax_price.plot(time, uncs1[3, :], label='price', color='g')
                ax_price.fill_between(x=time, y1=uncs1[3, :], color='gray', alpha=0.1)
                ax[i, j].legend()
                ax[i, j].set_xlabel('Hour')
                ax[i, j].set_ylabel(action_label[ai])
                ax[i, j].set_xlim((0, 23))
                ax[i, j].set_ylim((0, 1))
                ax_price.set_ylim((3.5, 13))
                ax_price.set_yticks([])


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(time, rollout_return1, color='r', linestyle='dashed', label=r'$\sum{l}$ with $\theta_0$')
    ax[0].fill_between(x=time, y1=rollout_return_spo1, label=r'$\sum{l_{\rm{spot}}}$', alpha=0.3)
    ax[0].fill_between(x=time, y1=rollout_return_temp1, label=r'$\sum{l_{\rm{temp}}}$', alpha=0.3)
    ax[0].legend()
    ax[0].set_xlabel('Hour')
    ax[1].plot(time, rollout_return2, color='r', linestyle='dashed', label=r'$\sum{l}$ with $\theta^{\star}$')
    ax[1].fill_between(x=time, y1=rollout_return_spo2, label=r'$\sum{l_{\rm{spot}}}$', alpha=0.3)
    ax[1].fill_between(x=time, y1=rollout_return_temp2, label=r'$\sum{l_{\rm{temp}}}$', alpha=0.3)
    ax[1].legend()
    ax[1].set_xlabel('Hour')

    plt.show()


#################
if __name__ == "__main__":
    ### read data and plot
    f_b = open('results_rl_initial_theta.pkl', "rb")
    results_b = pickle.load(f_b)
    f_b.close()
    f_a = open('results_rl_trained_theta.pkl', "rb")
    results_a = pickle.load(f_a)
    f_a.close()
    plot_data(results_b, results_a, state_dim=8, action_dim=6, n_step=96)
