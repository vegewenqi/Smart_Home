import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from gym.spaces.box import Box
import casadi as csd
import json
from base_types import Env


class SmartHome(Env):
    def __init__(self, env_params):
        # state box, dimension of 8
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 10, 10, 10, 0]),
            high=np.array([100, 100, 100, 100, 70, 70, 70, 5], dtype=np.float32),
        )
        # action box, dimension of 9
        self.action_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10, 10, 10, 10, 10, 10, 10, 5, 1], dtype=np.float32),
        )

        # adjustable env parameters in .json file
        # standard deviation
        self.epsilon_rad_sigma = env_params[
            "epsilon_rad_sigma"
        ]  # p_rad ~= 0.55 kW, sigma ~= 0.0001
        self.epsilon_out_sigma = env_params[
            "epsilon_out_sigma"
        ]  # T_out ~= 23, sigma ~= 0.1
        self.epsilon_app_sigma = env_params[
            "epsilon_app_sigma"
        ]  # p_app ~= 3.5 kW, sigma ~= 0.01
        self.epsilon_1234_sigma = env_params[
            "epsilon_1234_sigma"
        ]  # T ~= 40, sima ~= 0.1
        self.c_low = env_params["c_low"]
        self.c_hig = env_params["c_hig"]
        self.dt = env_params["dt"]  # 15min

        # uncertainty and price
        # data = pd.read_excel("Data/SmartHome/smart_home_uncertainties.ods", dtype=float)
        data = pd.read_excel("C:/Users/Administrator/Desktop/Safe-RL/Data/SmartHome/smart_home_uncertainties.ods",
                             dtype=float)
        self.uncertainty = np.concatenate(
            [
                data.p_rad.to_numpy()[None, :],
                data.p_app.to_numpy()[None, :],
                data.t_out.to_numpy()[None, :],
            ],
            axis=0,
        )
        # data = pd.read_excel("Data/SmartHome/smart_home_prices.ods", dtype=float)
        data = pd.read_excel("C:/Users/Administrator/Desktop/Safe-RL/Data/SmartHome/smart_home_prices.ods",
                             dtype=float)
        self.price = np.concatenate(
            [
                data.price_buy.to_numpy()[None, :],
                data.price_sell.to_numpy()[None, :],
            ],
            axis=0,
        )
        # self.uncertainty = np.ones((3, 10000))  # [p_rad, p_app, t_out]
        # self.price = np.ones((2, 10000))  # [price_buy, price_sell]

        # time
        self.t = 0

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
        self.A_pv = 35
        self.eta = 0.9

        self.state = None
        # initialization
        self.reset()

    # dimension=8, 9, 3
    def conti_model(self, state, action, uncertainty):
        # states
        T_w = state[0]
        T_in = state[1]
        T_g = state[2]
        T_p = state[3]
        T_1 = state[4]
        T_2 = state[5]
        T_3 = state[6]
        E = state[7]

        # inputs
        P_grid_pv = action[0]
        P_lod_pv = action[1]
        P_pv_ch = action[2]
        P_grid_ch = action[3]
        P_grid_dis = action[4]
        P_lod_dis = action[5]
        P_lod_grid = action[6]
        P_hp = action[7]
        X_v = action[8]

        # uncertainties
        P_rad = uncertainty[0]
        P_app = uncertainty[1]
        T_out = uncertainty[2]

        # others
        T_ret = (1 - math.exp(-self.rho)) * T_g + math.exp(-self.rho) * T_p
        T_inl = X_v * (T_1 - T_ret) + T_ret
        COP = self.a_cop * T_out + self.b_cop * (0.5 * (T_2 + T_3)) + self.c_cop
        P_ch = P_grid_ch + P_pv_ch
        P_dis = P_grid_dis + P_lod_dis

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
        d_E = self.eta * P_ch - 1 / self.eta * P_dis
        dot_state = np.array([d_T_w, d_T_in, d_T_g, d_T_p, d_T_1, d_T_2, d_T_3, d_E])
        return dot_state

    def discrete_model_real(self, state, action, uncertainty):
        # with noise
        k1 = self.conti_model(state, action, uncertainty)
        k2 = self.conti_model(state + 0.5 * self.dt * k1, action, uncertainty)
        k3 = self.conti_model(state + 0.5 * self.dt * k2, action, uncertainty)
        k4 = self.conti_model(state + self.dt * k3, action, uncertainty)
        next_state = (
            state
            + (1 / 6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
            + np.array(
                [
                    np.random.normal(0, self.epsilon_1234_sigma),
                    np.random.normal(0, self.epsilon_1234_sigma),
                    np.random.normal(0, self.epsilon_1234_sigma),
                    np.random.normal(0, self.epsilon_1234_sigma),
                    0,
                    0,
                    0,
                    0,
                ]
            )
        )
        return next_state

    # symbolic function
    def conti_model_sym(self, state: csd.SX, action: csd.SX, uncertainty: csd.SX):
        # states
        T_w = state[0]
        T_in = state[1]
        T_g = state[2]
        T_p = state[3]
        T_1 = state[4]
        T_2 = state[5]
        T_3 = state[6]
        E = state[7]

        # inputs
        P_grid_pv = action[0]
        P_lod_pv = action[1]
        P_pv_ch = action[2]
        P_grid_ch = action[3]
        P_grid_dis = action[4]
        P_lod_dis = action[5]
        P_lod_grid = action[6]
        P_hp = action[7]
        X_v = action[8]

        # uncertainties
        P_rad = uncertainty[0]
        P_app = uncertainty[1]
        T_out = uncertainty[2]

        # others
        T_ret = (1 - math.exp(-self.rho)) * T_g + math.exp(-self.rho) * T_p
        T_inl = X_v * (T_1 - T_ret) + T_ret
        COP = self.a_cop * T_out + self.b_cop * (0.5 * (T_2 + T_3)) + self.c_cop
        P_ch = P_grid_ch + P_pv_ch
        P_dis = P_grid_dis + P_lod_dis

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
        d_E = self.eta * P_ch - 1 / self.eta * P_dis
        dot_state = csd.vertcat(d_T_w, d_T_in, d_T_g, d_T_p, d_T_1, d_T_2, d_T_3, d_E)
        return dot_state

    # with theta, dimension of theta_model is 1*4
    # symbolic function
    def discrete_model_mpc(self, state, action, uncertainty, theta_model):
        k1 = self.conti_model_sym(state, action, uncertainty) + csd.vertcat(
            theta_model, csd.SX.zeros(4, 1)
        )
        k2 = self.conti_model_sym(state + 0.5 * self.dt * k1, action, uncertainty)
        k3 = self.conti_model_sym(state + 0.5 * self.dt * k2, action, uncertainty)
        k4 = self.conti_model_sym(state + self.dt * k3, action, uncertainty)
        next_state = state + (1 / 6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state

    # def get_model(self, state, action, uncertainty, theta_model):
    #     mpc_model = csd.Function("mpc_model", [state, action, uncertainty, theta_model],
    #                              self.discrete_model_mpc(state, action, uncertainty, theta_model))
    #     return mpc_model

    def reset(self):
        self.state = np.array([15, 25, 15, 27, 38, 50, 16, 3]) + np.array(
            [1, 0.05, 1, 1, 1, 0.1, 1, 0.1]
        ) * (np.random.normal(scale=1, size=8))
        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        self.t = 0
        return self.state

    def step(self, action):
        # real sys uncertainty = baseline estimate data + noises
        self.state = self.discrete_model_real(
            self.state,
            action,
            self.uncertainty[:, self.t]
            + np.array(
                [
                    np.random.normal(0, self.epsilon_rad_sigma),
                    np.random.normal(0, self.epsilon_app_sigma),
                    np.random.normal(0, self.epsilon_out_sigma),
                ]
            ),
        )
        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        rew, done = self.reward_fn(self.state, action, self.price[:, self.t])
        info = ""
        # dt = 15min, as 1/4 h
        self.t += 1
        return (self.state, rew, done, info)

    # return r, done
    def reward_fn(self, state, action, price):
        price_buy = price[0]
        price_sell = price[1]

        T_w = state[0]
        T_in = state[1]
        T_g = state[2]
        T_p = state[3]
        T_1 = state[4]
        T_2 = state[5]
        T_3 = state[6]
        E = state[7]

        P_grid_pv = action[0]
        P_lod_pv = action[1]
        P_pv_ch = action[2]
        P_grid_ch = action[3]
        P_grid_dis = action[4]
        P_lod_dis = action[5]
        P_lod_grid = action[6]
        P_hp = action[7]
        X_v = action[8]

        P_buy = P_grid_ch + P_lod_grid
        P_sell = P_grid_dis + P_grid_pv

        l_spo = price_buy * P_buy - price_sell * P_sell
        l_term = self.c_low * max((23 - T_in), 0) + self.c_hig * max((T_in - 27), 0)
        r = l_spo + l_term

        done = 0
        return r, done

    # not in use
    def cost_fn(self, state, action, next_state):
        pass


if __name__ == "__main__":
    with open('C:/Users/Administrator/Desktop/Safe-RL/Settings/other/smarthome_rl_lstd_mpc.json', 'r') as f:
        params = json.load(f)
        print(params)
    a = SmartHome(params["env_params"])
    print(a.state)
    print("---------")
    a.step([0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0.5])
    print(a.state)
