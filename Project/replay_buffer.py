import random
import numpy as np
from collections import deque


class BasicBuffer:
    def __init__(self, max_size, seed):
        self.max_size = max_size
        self.size = 0
        self.buffer = deque(maxlen=max_size)
        self.rng1 = np.random.default_rng(seed)
        self.rng2 = random.Random(seed)

    def push(self, state, obs, action, reward, next_state, next_obs, done, info):
        experience = (state, obs, action, np.array([reward]), next_state, next_obs, done, info)
        self.size = min(self.size+1, self.max_size)
        self.buffer.append(experience)

    def sample(self, batch_size, last_roll=False):
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        batch = self.rng2.sample(self.buffer, batch_size)

        for experience in batch:
            state, obs, action, reward, next_state, next_obs, done, info = experience
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)
        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def sample_sequence(self, batch_size):
        # batch_size is taken to be the size of each episode
        ## ToDo: seeding
        state_batch = []
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_obs_batch = []
        done_batch = []
        info_batch = []

        max_episodes = len(self.buffer) // batch_size
        start = np.random.randint(0, max_episodes)

        for sample in range(start * batch_size, (start + 1) * batch_size):
            state, obs, action, reward, next_state, next_obs, done, info = self.buffer[sample]
            # state, action, reward, next_state, done = experience
            state_batch.append(state)
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
            info_batch.append(info)

        return (
            state_batch,
            obs_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_obs_batch,
            done_batch,
            info_batch,
        )

    def __len__(self):
        return len(self.buffer)
