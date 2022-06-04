import numpy as np
import pandas as pd
from helpers import tqdm_context
from Agents.abstract_agent import TrainableController


def rollout_sample(env, agent, replay_buffer, n_step, mode="train"):
    trained_theta = np.array([-3.90143530e-01, -3.90268995e-01, -3.90063883e-01, -3.90031666e-01, 5.38572361e+00,
                              3.90385042e-01, 3.90385042e-01, 3.00419620e+02, 3.41867374e+00, -1.40794169e-01,
                              3.32972969e-01, 1.85456924e-01, -2.52337309e-01, 1.00000000e-01, 2.32853000e-03,
                              -4.22412097e-01, -3.93204979e-01, 1.00000000e-01, -1.08394830e-02])[:, None]

    state = env.reset()

    rollout_return = 0
    rollout_return_spo = 0
    rollout_return_temp = 0
    states = np.zeros((agent.obs_dim, n_step))
    actions = np.zeros((agent.action_dim, n_step))
    uncs = np.zeros((4, n_step))
    rollout_returns = np.zeros((1, n_step))
    rollout_returns_spo = np.zeros((1, n_step))
    rollout_returns_temp = np.zeros((1, n_step))
    for step in tqdm_context(range(n_step), desc="Episode", pos=1):
        print(f'evaluation step {step}-------------')
        # action, add_info = agent.get_action(state, time=env.t, mode=mode)
        action, add_info = agent.get_action(state, act_wt=trained_theta, time=env.t, mode=mode)
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
