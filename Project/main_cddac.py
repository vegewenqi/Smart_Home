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

# cmd_line = sys.argv
# with open(cmd_line[1]) as f:
#     params = json.load(f)
#     print(params)

json_path = os.path.abspath(os.path.join(os.getcwd(), '../Settings/other/smarthome_rl_mpc_cddac.json'))
with open(json_path, 'r') as f:
    params = json.load(f)
    params["env_params"]["json_path"] = json_path
    print(f'env_params = {params}')

### Environment
env_train = env_init(params["env"], params["env_params"])
env_evaluate = env_init(params["env"], params["env_params"])

### Agent
agent = agent_init(env_train, params["agent"], params["agent_params"])
print(f'initial theta = {agent.actor.actor_wt.squeeze()}')

### Reply buffer
replay_buffer = BasicBuffer(params["buffer_maxlen"])

### Returns
theta_log = []
eval_returns = []
for it in tqdm_context(range(params["n_iterations"]), desc="Iterations", pos=3):
    print(f"Iteration: {it}------")
    # Randomly initialize the starting state
    state = env_train.reset()
    for step in tqdm_context(range(params["epi_length"]), desc="Episode", pos=1):
        print(f'Iteration {it} epi_step {step}-------------')
        action, add_info = agent.get_action(state, mode="train")
        next_state, reward, done = env_train.step(action)
        replay_buffer.push(state, action, reward, next_state, done, add_info)
        state = next_state

        # Train step
        train_controller(agent, replay_buffer)
        theta_log.append(agent.actor.actor_wt)

        # Evaluation
        if (len(replay_buffer) >= agent.batch_size) and ((step+1) % params["eval_delay"] == 0):
            eval_return = []
            for eval_runs in tqdm_context(range(params["n_evals"]), desc="Evaluation Rollouts"):
                rollout_return = rollout_sample(env_evaluate, agent, replay_buffer, params["epi_length"], mode="eval")
            eval_return.append(rollout_return)
            eval_returns.append(mean(eval_return))
            # print(f"Evaluation return: {mean(eval_return)}")
            print(f"Evaluation returns: {eval_returns}")


### save results
Results = {'theta_log': theta_log,
           'eval_returns': eval_returns}
f = open('Results/SmartHome/results_rl.pkl', "wb")
pickle.dump(Results, f)
f.close()
print('Results saved successfully！')


### Final rollout for visualization
# _ = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="final")
# # Evaluation performance plot
# perf = plt.figure("Evaluation Performance")
# plt.plot(eval_returns)
# plt.ylabel("return")
# plt.show()
# print(eval_returns)
# plt.pause(5)



