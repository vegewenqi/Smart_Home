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
from rollout_utils_rl_test import rollout_sample, train_controller
import pickle

# cmd_line = sys.argv
# with open(cmd_line[1]) as f:
#     params = json.load(f)
#     print(params)

json_path = os.path.abspath(os.path.join(os.getcwd(), '../Settings/other/smarthome_rl_mpc_test.json'))
with open(json_path, 'r') as f:
    params = json.load(f)
    params["env_params"]["json_path"] = json_path
    print(f'env_params = {params}')

### Environment
env = env_init(params["env"], params["env_params"])

### Agent
agent = agent_init(env, params["agent"], params["agent_params"])

### Reply buffer
replay_buffer = BasicBuffer(params["buffer_maxlen"])

# Evaluation
states, actions, rollout_returns, rollout_returns_spo, rollout_returns_temp, uncs = rollout_sample(
    env, agent, replay_buffer, params["epi_length"], mode="eval")

data_set = np.concatenate((states, actions, rollout_returns, rollout_returns_spo, rollout_returns_temp, uncs))

# # # Average return
# Return = []
# for eval_runs in tqdm_context(range(params["n_evals"]), desc="Evaluation Rollouts"):
#     states, actions, rollout_returns, rollout_returns_spo, rollout_returns_temp, uncs = rollout_sample(
#         env, agent, replay_buffer, params["epi_length"], mode="eval")
#     data_set = Return.append(rollout_returns[0, -1])
# Return = np.mean(Return)

### save results
Results = data_set
f = open('Results/SmartHome/results_rl_trained_theta33.pkl', "wb")
# f = open('Results/SmartHome/results_rl_initial_theta.pkl', "wb")
pickle.dump(Results, f)
f.close()
print('Results saved successfully！')




