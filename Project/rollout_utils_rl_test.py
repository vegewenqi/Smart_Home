import numpy as np
import pandas as pd
from helpers import tqdm_context
from Agents.abstract_agent import TrainableController


def rollout_sample(env, agent, replay_buffer, n_step, mode="eval"):
    trained_theta = np.array([1.00219891e+00, 1.00237787e+00, 9.97582204e-01, 9.99608593e-01, 1.00143571e+00,
                              1.00032285e+00, 1.94246887e-02, -4.59972079e-03, 5.00104884e+00, 2.51232104e-03,
                              2.51232104e-03, 2.99434724e+01, 2.94344886e+00, 0.00000000e+00, 5.86117806e-04])[:, None]

    state = env.reset()
    state_4 = env.extract_state(state)

    rollout_return = 0
    rollout_return_spo = 0
    rollout_return_temp = 0
    states = np.zeros((agent.obs_dim+4, n_step))
    actions = np.zeros((agent.action_dim, n_step))
    uncs = np.zeros((4, n_step))
    rollout_returns = np.zeros((1, n_step))
    rollout_returns_spo = np.zeros((1, n_step))
    rollout_returns_temp = np.zeros((1, n_step))
    for step in tqdm_context(range(n_step), desc="Episode", pos=1):
        print(f'evaluation step {step}-------------')
        # action, add_info = agent.get_action(state_4, time=env.t, mode=mode)
        action, add_info = agent.get_action(state_4, act_wt=trained_theta, time=env.t, mode=mode)
        next_state, reward, done, l_spo, l_temp, unc_t = env.step(action)

        states[:, step] = state
        actions[:, step] = action
        Price_t = env.price[0, add_info['time']]
        uncs[:, step] = np.concatenate((unc_t, Price_t), axis=None)

        # if mode == "train" or mode == "mpc":
        #     replay_buffer.push(state, action, reward, next_state, done, add_info)

        rollout_return += reward
        rollout_return_spo += l_spo
        rollout_return_temp += l_temp
        rollout_returns[:, step] = rollout_return
        rollout_returns_spo[:, step] = rollout_return_spo
        rollout_returns_temp[:, step] = rollout_return_temp

        # state = next_state.copy()
        state = next_state
        state_4 = env.extract_state(next_state)

    return states, actions, rollout_returns, rollout_returns_spo, rollout_returns_temp, uncs


# execute agent.train:
#  run replay_buffer.sample function:
#     sample batch_size data
#     then use those data to calculate w,v, and then theta
#     update self.critic_wt, self.adv_wt, self.actor.actor_wt
def train_controller(agent, replay_buffer):
    if isinstance(agent, TrainableController):
        # train_it = min(
        #     agent.iterations, int(5 * len(replay_buffer.buffer) / agent.batch_size)
        # )
        train_it = agent.iterations
        if len(replay_buffer) > agent.batch_size:
            print('train')
            agent.train(replay_buffer, train_it)


# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in many_episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in many_episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in many_episode_rewards]

    return minimum, maximum, mean
