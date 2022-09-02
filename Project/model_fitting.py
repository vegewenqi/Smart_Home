import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import os
from Environments import env_init
from Agents import agent_init
from replay_buffer import BasicBuffer
from helpers import tqdm_context
from rollout_utils import rollout_sample, train_controller
import pickle
import casadi as csd
import pandas as pd
import seaborn as sns


#####-------------------collect data fitting data from the real system-------------------
# # cmd_line = sys.argv
# # with open(cmd_line[1]) as f:
# #     params = json.load(f)
# #     print(params)
#
# json_path = os.path.abspath(os.path.join(os.getcwd(), '../Settings/other/smarthome_rl_mpc_model_fitting.json'))
# with open(json_path, 'r') as f:
#     params = json.load(f)
#     params["env_params"]["json_path"] = json_path
#     print(f'env_params = {params}')
#
# ### Environment
# env = env_init(params["env"], params["env_params"])
#
# ### Reply buffer
# replay_buffer = BasicBuffer(params["buffer_maxlen"])
#
# ### Returns
# returns = []
#
# for it in tqdm_context(range(params["n_iterations"]), desc="Iterations", pos=3):
#     # print(f"Iteration: {it}------")
#     # Randomly initialize the starting state
#     rollout_return = 0
#     state = env.reset()
#     for step in tqdm_context(range(params["epi_length"]), desc="Episode", pos=1):
#         # print(f'Iteration {it} epi_step {step}-------------')
#         action = np.array([np.random.rand(), np.random.rand(), 5*np.random.rand(),
#                            5*np.random.rand(), 3*np.random.rand(), 0.2+0.6*np.random.rand()])
#
#         case = np.random.randint(4)
#         if case == 0:
#             action[[0, 2]] = 0
#         elif case == 1:
#             action[[0, 3]] = 0
#         elif case == 2:
#             action[[1, 2]] = 0
#         else:
#             action[[1, 3]] = 0
#
#         next_state, reward, done = env.step(action)
#         replay_buffer.push(state, action, reward, next_state, done, env.t)
#
#         state = next_state
#         rollout_return += reward
#     returns.append(rollout_return)
#
# ### save results
# Results = {'returns': returns,
#            'buffer': replay_buffer}
# f = open('Results/SmartHome/fitting_data.pkl', "wb")
# pickle.dump(Results, f)
# f.close()
# print('Results saved successfully！')
# #
#

#####----------------fitting process---------------------------
# f = open('Results/SmartHome/fitting_data.pkl', "rb")
# results = pickle.load(f)
# f.close()
# buffer_data = results['buffer'].buffer
#
# json_path = os.path.abspath(os.path.join(os.getcwd(), '../Settings/other/smarthome_rl_mpc_model_fitting.json'))
# with open(json_path, 'r') as f:
#     params = json.load(f)
#     params["env_params"]["json_path"] = json_path
#     print(f'env_params = {params}')
# env_smp = env_init(params["env"], params["env_params"])
#
# S = csd.MX.sym('state', 4, 960)
# A = csd.MX.sym('action', 6, 960)
# UNC = csd.MX.sym('uncertainty', 3, 960)
# theta_model = csd.MX.sym("theta_model", 8)
# S_P = csd.MX.sym('next state', 4, 960)
# error = 0
# for i in range(960):
#     next_state = env_smp.discrete_model_mpc(S[:, i], A[:, i], UNC[:, i], theta_model)
#     a = S_P[:, i] - next_state
#     error += csd.norm_2(S_P[:, i] - next_state)
# ErrorFunc = csd.Function("Error", [csd.reshape(S, -1, 1), csd.reshape(A, -1, 1), csd.reshape(UNC, -1, 1),
#                                    theta_model, csd.reshape(S_P, -1, 1)], [error])
# grad = ErrorFunc.factory("grad", ["i0", "i1", "i2", "i3", "i4"], ["jac:o0:i3"])
#
# fitting_error = []
# fitting_ite = []
# for tt in range(5):
#     theta_model_value = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
#     for iteration in range(100):
#         state = np.zeros((4, 960))
#         action = np.zeros((6, 960))
#         uncertainty = np.zeros((3, 960))
#         next_state_real = np.zeros((4, 960))
#         for i in range(960):
#             state[:, i] = env_smp.extract_state(buffer_data[i + iteration * 960 + tt*96000][0])
#             action[:, i] = buffer_data[i + iteration * 960 + tt*96000][1]
#             uncertainty[:, i] = env_smp.uncertainty[:, buffer_data[i + iteration * 960 + tt*96000][5]]
#             next_state_real[:, i] = env_smp.extract_state(buffer_data[i + iteration * 960 + tt*96000][3])
#         state = np.reshape(state, (-1, 1), order='F')
#         action = np.reshape(action, (-1, 1), order='F')
#         uncertainty = np.reshape(uncertainty, (-1, 1), order='F')
#         next_state_real = np.reshape(next_state_real, (-1, 1), order='F')
#         error_train = ErrorFunc(state, action, uncertainty, theta_model_value, next_state_real).full()[0, 0]
#         theta_model_value -= 0.000000001 * grad(state, action, uncertainty, theta_model_value,
#                                                 next_state_real).full().squeeze()
#         print(error_train)
#         print(theta_model_value)
#         print(iteration)
#         fitting_error.append(error_train)
#         fitting_ite.append(iteration)
# Results = {'fitting_error': fitting_error,
#            'fitting_ite': fitting_ite}
#
# f = open('Results/SmartHome/results_fitting_error.pkl', "wb")
# pickle.dump(Results, f)
# f.close()
# print('Results saved successfully！')
#

########-------------------plotting the fitting error---------------------
# sns.set_theme(style="darkgrid")
# f = open('Results/SmartHome/results_fitting_error.pkl', "rb")
# results = pickle.load(f)
# f.close()
# df = pd.DataFrame(dict(Steps=results['fitting_ite'],
#                        Error=results['fitting_error']))
# g = sns.relplot(x="Steps", y="Error", kind="line", data=df)
# g.set(xlabel="Steps", ylabel="Fitting error")
# plt.xlim((0, 100))
# # plt.legend(labels=["Mean Squared Fitting Error", "95% Confidence interval"])
# plt.show(block=True)


########-------------------plotting the perofrmance bar for three cases of theta---------------------
sns.set_theme(style="whitegrid")
df = pd.DataFrame(dict(Theta=[r"$\theta_{\rm{mi}}$", r"$\theta_{\rm{mi}}$", r"$\theta_{\rm{mi}}$", r"$\theta_0$",
                              r"$\theta_0$", r"$\theta_0$", r"$\theta_{\star}$", r"$\theta_{\star}$",
                              r"$\theta_{\star}$"],
                       Performance=[7445, 7660, 7101, 1303, 1200, 1145, 104, 110, 98]))
g = sns.barplot(x="Theta", y="Performance", data=df)
g.set(xlabel=None, ylabel=r"Performance")
g.bar_label(g.containers[0])
plt.show(block=True)

