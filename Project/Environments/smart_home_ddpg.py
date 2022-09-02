import numpy as np
import pandas as pd
import math
from gym.spaces.box import Box
import json
from base_types import Env
import os


class SmartHome(Env):
    def __init__(self, env_params={}):
        # state box, dimension of 9 (8 system states + 1 time state)
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 10, 10, 10, 0, 0]),
            high=np.array([100, 100, 100, 100, 70, 70, 70, 5, 96], dtype=np.float32),
        )
        # action box, dimension of 9
        self.action_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            # high=np.array([10, 10, 10, 10, 10, 1], dtype=np.float32),
            high=np.array([1.2, 1.2, 6, 6, 3.6, 1], dtype=np.float32),
        )

        # standard deviation
        self.epsilon_rad_sigma = env_params["epsilon_rad_sigma"]  # p_rad ~= 0.55 kW, sigma ~= 0.0001
        self.epsilon_out_sigma = env_params["epsilon_out_sigma"]  # T_out ~= 23, sigma ~= 0.1
        self.epsilon_app_sigma = env_params["epsilon_app_sigma"]  # p_app ~= 3.5 kW, sigma ~= 0.01
        self.epsilon_1234_sigma = env_params["epsilon_1234_sigma"]  # T ~= 40, sima ~= 0.1
        self.c_low = env_params["c_low"]
        self.c_hig = env_params["c_hig"]
        self.dt = env_params["dt"]  # dt=900s=15min

        # uncertainty and price
        SafeRL_path = os.path.abspath(os.path.join(os.getcwd(), '../'))

        data_path = os.path.join(SafeRL_path, 'Data/SmartHome/smart_home_uncertainties.ods')
        data = pd.read_excel(data_path, dtype=float)
        self.uncertainty = np.concatenate(
            [
                data.p_rad.to_numpy()[None, :],
                data.p_app.to_numpy()[None, :],
                data.t_out.to_numpy()[None, :],
            ],
            axis=0,
        )
        self.uncertainty = np.repeat(self.uncertainty, 4, axis=1)
        # print(self.uncertainty.shape)

        data_path = os.path.join(SafeRL_path, 'Data/SmartHome/smart_home_prices.ods')
        data = pd.read_excel(data_path, dtype=float)
        self.price = np.concatenate(
            [
                data.price_buy.to_numpy()[None, :],
                data.price_sell.to_numpy()[None, :],
            ],
            axis=0,
        )
        self.price = np.repeat(self.price, 4, axis=1)
        # self.uncertainty = np.ones((3, 10000))  # [p_rad, p_app, t_out]
        # self.price = np.ones((2, 10000))  # [price_buy, price_sell]

        # system parameters
        self.k_w_out = 64.8
        self.k_w_in = 64.8
        self.k_g_in = 594.8
        self.k_p_g = 506.2
        self.C_w = 312 * 10 ** 4
        self.C_in = 4.4 * 10 ** 6
        self.C_g = 18 * 10 ** 5
        self.C_p = 1.7 * 10 ** 6
        self.R_w = 0.99
        self.M_inl = 0.062
        self.C_wat = 4180
        self.R_1 = 18.8
        self.R_2 = 18.8
        self.R_3 = 18.8
        self.rho = 7.5
        self.m_1 = 66.38
        self.m_2 = 66.38
        self.m_3 = 66.38
        self.a_cop = 0.088
        self.b_cop = -0.079
        self.c_cop = 7.253
        self.beta = 0.429
        self.A_pv = 35  # 35m*2
        self.eta = 0.9

        self.state = np.zeros(9)
        # initialization
        self.reset()

    # dimension=9, 6
    def conti_model(self, state, action):
        # states
        T_w = state[0]
        T_in = state[1]
        T_g = state[2]
        T_p = state[3]
        T_1 = state[4]
        T_2 = state[5]
        T_3 = state[6]
        E = state[7]
        time = int(state[8])

        # inputs
        P_ch = action[0]
        P_dis = action[1]
        P_buy = action[2]
        P_sell = action[3]
        P_hp = action[4]
        X_v = action[5]

        # real uncertainties data + prediction errors
        P_rad = self.uncertainty[0, time] + np.random.normal(0, self.epsilon_rad_sigma)
        P_app = self.uncertainty[1, time] + np.random.normal(0, self.epsilon_app_sigma)
        T_out = self.uncertainty[2, time] + np.random.normal(0, self.epsilon_out_sigma)

        # others
        T_ret = (1 - math.exp(-self.rho)) * T_g + math.exp(-self.rho) * T_p
        T_inl = X_v * (T_1 - T_ret) + T_ret
        COP = self.a_cop * T_out + self.b_cop * (0.5 * (T_2 + T_3)) + self.c_cop

        # ODEs
        d_T_w = (
                1 / self.C_w * (self.k_w_out * (T_out - T_w) + self.k_w_in * (T_in - T_w))
        )
        d_T_in = (
                1 / self.C_in * (self.k_w_in * (T_w - T_in) + self.k_g_in * (T_g - T_in))
        )
        d_T_g = 1 / self.C_g * (self.k_g_in * (T_in - T_g) + self.k_p_g * (T_p - T_g))
        d_T_p = (
                1
                / self.C_p
                * (self.k_p_g * (T_g - T_p) + self.M_inl * self.C_wat * (T_inl - T_p))
        )
        d_T_1 = (
                1
                / (self.m_1 * self.C_wat)
                * (
                        self.R_1 * (T_2 - T_1)
                        - self.R_w * (T_1 - T_out)
                        + X_v * self.M_inl * self.C_wat * (T_2 - T_1)
                )
        )
        d_T_2 = (
                1
                / (self.m_2 * self.C_wat)
                * (
                        self.R_2 * (T_3 - T_2)
                        - self.R_2 * (T_2 - T_1)
                        - self.R_w * (T_2 - T_out)
                        + X_v * self.M_inl * self.C_wat * (T_3 - T_2)
                        + COP * 1000 * P_hp
                )
        )
        d_T_3 = (
                1
                / (self.m_3 * self.C_wat)
                * (
                        -self.R_3 * (T_3 - T_2)
                        - self.R_w * (T_3 - T_out)
                        + X_v * self.M_inl * self.C_wat * (T_ret - T_3)
                )
        )
        d_E = 1 / 3600 * (self.eta * P_ch - 1 / self.eta * P_dis)  # battery's sampling time is one hour

        dot_state = np.array([d_T_w, d_T_in, d_T_g, d_T_p, d_T_1, d_T_2, d_T_3, d_E, 0])  # d_time=0
        return dot_state

    def discrete_model_real(self, state, action):
        # with noise
        k1 = self.conti_model(state, action)
        k2 = self.conti_model(state + 0.5 * self.dt * k1, action)
        k3 = self.conti_model(state + 0.5 * self.dt * k2, action)
        k4 = self.conti_model(state + self.dt * k3, action)
        next_state = (state
                      + (1 / 6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
                      + np.array([np.random.normal(0, self.epsilon_1234_sigma),
                                  np.random.normal(0, self.epsilon_1234_sigma),
                                  np.random.normal(0, self.epsilon_1234_sigma),
                                  np.random.normal(0, self.epsilon_1234_sigma),
                                  0,
                                  0,
                                  0,
                                  0,
                                  0]))
        next_state[8] = state[8] + 1  # time step forward
        return next_state

    def reset(self):
        self.state[:8] = np.array([15, 25, 15, 27, 38, 50, 16, 2]) + 0.1 * np.array(
            [1, 0.05, 1, 1, 1, 0.1, 1, 0.1]
        ) * (np.random.normal(scale=1, size=8))
        # self.state = np.array([15, 25, 15, 27, 38, 50, 16, 2]) + 0.0 * np.array(
        #     [1, 0.05, 1, 1, 1, 0.1, 1, 0.1]
        # ) * (np.random.normal(scale=1, size=8))

        self.state[8] = 0   # reset time as zero

        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        return self.state

    # def extract_state(self, state_8):
    #     state_4 = np.array([state_8[1], state_8[2], state_8[5], state_8[7]])
    #     return state_4

    def step(self, action):
        # real sys uncertainty = baseline estimate data + noises
        self.state = self.discrete_model_real(self.state, action)
        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        rew, done = self.reward_fn(self.state, action)
        return self.state, rew, done

    # return r, done
    def reward_fn(self, state, action):
        T_w = state[0]
        T_in = state[1]
        T_g = state[2]
        T_p = state[3]
        T_1 = state[4]
        T_2 = state[5]
        T_3 = state[6]
        E = state[7]
        time = int(state[8])

        P_ch = action[0]
        P_dis = action[1]
        P_buy = action[2]
        P_sell = action[3]
        P_hp = action[4]
        X_v = action[5]

        price_buy = self.price[0, time]
        price_sell = self.price[1, time]

        l_spo = price_buy * P_buy - price_sell * P_sell
        l_temp = self.c_low * max((23 - T_in), 0) + self.c_hig * max((T_in - 27), 0)
        # l_temp = self.c_low * (T_in - 23) * (T_in - 27) + self.c_low * 4
        r = l_spo + l_temp

        done = False
        if time == 96:
            done = True
        return r, done

    # not in use
    def cost_fn(self, state, action, next_state):
        pass


if __name__ == "__main__":
    print(os.getcwd())
    with open('../../Settings/other/smarthome_rl_mpc_cddac_fast_try.json', 'r') as f:
        params = json.load(f)
        print(params)
    a = SmartHome(params["env_params"])
    print(a.observation_space.shape)
    print(a.action_space.shape[0])
    a.reset()
    print(a.state)
    # print("---------")
    # a.step([0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0.5])
    # print(a.state)
