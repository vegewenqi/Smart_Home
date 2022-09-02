from helpers import tqdm_context
import numpy as np
import pandas as pd
import os
import math
from gym.spaces.box import Box
import casadi as csd
import json
from base_types import Env
from Project.Results.SmartHome.plot_mpc import save_data


def rollout_sample(env, agent, n_step, mode="train"):
    states = np.zeros((agent.obs_dim, n_step))
    actions = np.zeros((agent.action_dim + 3, n_step))
    l_spos = np.zeros((1, n_step))
    l_tems = np.zeros((1, n_step))
    rewards = np.zeros((1, n_step))
    rollout_return = 0
    rollout_returns = np.zeros((1, n_step))

    state = env.reset()

    for step in tqdm_context(range(n_step), desc="Episode", pos=1):
        action, add_info = agent.get_action(state, mode=mode)
        next_state, reward, done, _ = env.step(action)

        # record P_pv and P_app for analysis
        P_pv = env.beta * env.A_pv * env.uncertainty[0, add_info['time']]
        P_app = env.uncertainty[1, add_info['time']]
        Price = env.price[0, add_info['time']]
        aug_action = np.concatenate((action, P_pv, P_app, Price), axis=None)

        states[:, step] = state
        actions[:, step] = aug_action

        # print(add_info["soln"]["f"].full())
        print(f'step: {step}')
        time = add_info["time"]
        l_spos[:, step] = l_spo_fn(action, env.price[:, time])
        l_tems[:, step] = l_tem_fn(next_state, env.c_low)
        rewards[:, step] = reward
        rollout_return += reward
        rollout_returns[:, step] = rollout_return

        state = next_state.copy()

    return states, actions, l_spos, l_tems, rewards, rollout_returns


def l_spo_fn(action, price):
    l_spo = price[0] * action[2] - price[1] * action[3]
    return l_spo


def l_tem_fn(state, coefficient):
    l_tem = coefficient * (state[1] - 23) * (state[1] - 27) + coefficient * 4
    return l_tem


class SmartHome(Env):
    def __init__(self, env_params={}):
        # state box, dimension of 8
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 10, 10, 10, 0]),
            high=np.array([100, 100, 100, 100, 70, 70, 70, 5], dtype=np.float32),
        )
        # action box, dimension of 9
        self.action_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([10, 10, 10, 10, 10, 1], dtype=np.float32),
        )

        # adjustable env parameters in .json file
        self.env_exist_noise = env_params["env_exist_noise"]
        # standard deviation
        self.epsilon_rad_sigma = env_params["epsilon_rad_sigma"]  # p_rad ~= 0.55 kW, sigma ~= 0.0001
        self.epsilon_out_sigma = env_params["epsilon_out_sigma"]  # T_out ~= 23, sigma ~= 0.1
        self.epsilon_app_sigma = env_params["epsilon_app_sigma"]  # p_app ~= 3.5 kW, sigma ~= 0.01
        self.epsilon_1234_sigma = env_params["epsilon_1234_sigma"]  # T ~= 40, sima ~= 0.1
        self.c_low = env_params["c_low"]
        self.c_hig = env_params["c_hig"]
        self.dt = env_params["dt"]  # dt=900s=15min

        # uncertainty and price
        SafeRL_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

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

        # time
        self.t = None

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

        self.state = None
        # initialization
        self.reset()

    # dimension=8, 6, 3
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
        P_ch = action[0]
        P_dis = action[1]
        P_buy = action[2]
        P_sell = action[3]
        P_hp = action[4]
        X_v = action[5]

        # uncertainties
        P_rad = uncertainty[0]
        P_app = uncertainty[1]
        T_out = uncertainty[2]

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
        dot_state = np.array([d_T_w, d_T_in, d_T_g, d_T_p, d_T_1, d_T_2, d_T_3, d_E])
        return dot_state

    def discrete_model_real(self, state, action, uncertainty):
        # with noise
        k1 = self.conti_model(state, action, uncertainty)
        k2 = self.conti_model(state + 0.5 * self.dt * k1, action, uncertainty)
        k3 = self.conti_model(state + 0.5 * self.dt * k2, action, uncertainty)
        k4 = self.conti_model(state + self.dt * k3, action, uncertainty)
        next_state = (state
                      + (1 / 6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
                      + self.env_exist_noise * np.array([np.random.normal(0, self.epsilon_1234_sigma),
                                                         np.random.normal(0, self.epsilon_1234_sigma),
                                                         np.random.normal(0, self.epsilon_1234_sigma),
                                                         np.random.normal(0, self.epsilon_1234_sigma),
                                                         0,
                                                         0,
                                                         0,
                                                         0]))
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
        P_ch = action[0]
        P_dis = action[1]
        P_buy = action[2]
        P_sell = action[3]
        P_hp = action[4]
        X_v = action[5]

        # uncertainties
        P_rad = uncertainty[0]
        P_app = uncertainty[1]
        T_out = uncertainty[2]

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
        d_E = 1 / 3600 * (self.eta * P_ch - 1 / self.eta * P_dis)
        dot_state = csd.vertcat(d_T_w, d_T_in, d_T_g, d_T_p, d_T_1, d_T_2, d_T_3, d_E)
        return dot_state

    # with theta, dimension of theta_model is 1*4
    # symbolic function
    def discrete_model_mpc(self, state: csd.SX, action: csd.SX, uncertainty: csd.SX, theta_model: csd.SX):
        k1 = self.conti_model_sym(state, action, uncertainty)
        k2 = self.conti_model_sym(state + 0.5 * self.dt * k1, action, uncertainty)
        k3 = self.conti_model_sym(state + 0.5 * self.dt * k2, action, uncertainty)
        k4 = self.conti_model_sym(state + self.dt * k3, action, uncertainty)
        next_state = state + (1 / 6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4) + csd.vertcat(
            theta_model, csd.SX.zeros(4, 1)
        )
        return next_state

    # def get_model(self, state, action, uncertainty, theta_model):
    #     mpc_model = csd.Function("mpc_model", [state, action, uncertainty, theta_model],
    #                              self.discrete_model_mpc(state, action, uncertainty, theta_model))
    #     return mpc_model

    def reset(self):
        self.state = np.array([15, 25, 15, 27, 38, 50, 16, 2]) + 0 * np.array(
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
            + self.env_exist_noise * np.array([np.random.normal(0, self.epsilon_rad_sigma),
                                               np.random.normal(0, self.epsilon_app_sigma),
                                               np.random.normal(0, self.epsilon_out_sigma)]))
        self.state = self.state.clip(
            self.observation_space.low, self.observation_space.high
        )
        rew, done = self.reward_fn(self.state, action, self.price[:, self.t])
        info = ""
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

        P_ch = action[0]
        P_dis = action[1]
        P_buy = action[2]
        P_sell = action[3]
        P_hp = action[4]
        X_v = action[5]

        l_spo = price_buy * P_buy - price_sell * P_sell
        # l_term = self.c_low * max((23 - T_in), 0) + self.c_hig * max((T_in - 27), 0)
        l_temp = self.c_low * (T_in - 23) * (T_in - 27) + self.c_low * 4
        r = l_spo + l_temp

        done = 0
        return r, done

    # not in use
    def cost_fn(self, state, action, next_state):
        pass


# RLMPC_LSTDQ_Agent that contains MPC actor, critics Q and V, train function
class Smart_Home_MPCAgent:
    def __init__(self, env, agent_params):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Hyper-parameters
        self._parse_agent_params(**agent_params)

        # Actor initialization
        self.actor = Custom_MPCActor(self.env, self.h, self.cost_params, self.gamma)

    # return act, info
    def get_action(self, state, act_wt=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0
        act, info = self.actor.act_forward(state, act_wt=act_wt, mode=mode)
        act += eps * (np.random.rand(self.action_dim))  # the added noise should de changed
        act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def _parse_agent_params(self, cost_params, eps, gamma, actor_lr, horizon, debug,
                            train_params, constrained_updates=True, experience_replay=False):
        self.cost_params = cost_params
        self.eps = eps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.h = horizon
        self.constrained_updates = constrained_updates
        self.experience_replay = experience_replay
        self.debug = debug
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]


# build symbolic nlps and symbolic sensitivities
class Custom_QP_formulation:
    def __init__(self, env, opt_horizon, gamma=1.0, th_param="custom", upper_tri=False):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.N = opt_horizon
        self.gamma = gamma
        self.etau = 1e-8
        self.th_param = th_param
        self.upper_tri = upper_tri

        self.beta = 0.429
        self.A_pv = 35  # 35m*2

        # Symbolic variables for optimization problem
        self.x = csd.SX.sym("x", self.obs_dim)
        self.u = csd.SX.sym("u", self.action_dim)
        self.X = csd.SX.sym("X", self.obs_dim, self.N)
        self.U = csd.SX.sym("U", self.action_dim, self.N)
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1),
            csd.reshape(self.X, -1, 1),
        )

        # Symbolic variables for all parameters: uncertainty, price, theta
        # uncertainty
        # dimension = 3 * N
        self.p_rad = csd.SX.sym('p_rad', 1)
        self.p_app = csd.SX.sym('p_app', 1)
        self.t_out = csd.SX.sym('p_out', 1)
        self.unc = csd.vertcat(self.p_rad, self.p_app, self.t_out)
        self.UNC = csd.SX.sym("UNC", self.unc.size()[0], self.N)
        self.UNC_dim = self.unc.size()[0] * self.N
        # price
        # dimension = 2 * N
        self.price_buy = csd.SX.sym('price_buy', 1)
        self.price_sell = csd.SX.sym('price_sell', 1)
        self.price = csd.vertcat(self.price_buy, self.price_sell)
        self.PRICE = csd.SX.sym("PRICE", self.price.size()[0], self.N)
        self.PRICE_dim = self.price.size()[0] * self.N
        # theta
        # dimension = 20
        self.theta_model = csd.SX.sym("theta_model", 4)
        self.theta_cop = csd.SX.sym("theta_cop", 1)
        self.theta_in = csd.SX.sym("theta_in", 3)
        self.theta_en = csd.SX.sym("theta_en", 2)
        self.theta_t = csd.SX.sym("theta_t", 3)
        self.theta_hp = csd.SX.sym("theta_hp", 1)
        self.theta_xv = csd.SX.sym("theta_xv", 1)
        self.theta_e = csd.SX.sym("theta_e", 1)
        self.theta_ch_dis = csd.SX.sym("theta_ch_dis", 2)
        self.theta_buy_sell = csd.SX.sym("theta_buy_sell", 2)
        self.theta = csd.vertcat(self.theta_model, self.theta_cop, self.theta_in, self.theta_en, self.theta_t,
                                 self.theta_hp, self.theta_xv, self.theta_e, self.theta_ch_dis, self.theta_buy_sell)
        self.theta_dim = self.theta.size()[0]

        # [Initial state=8, theta params=20, uncertainty=3*N, price=2*N]
        self.P = csd.vertcat(self.x, self.theta, csd.reshape(self.UNC, -1, 1), csd.reshape(self.PRICE, -1, 1))
        # note that csd.reshape() is reshaped by column
        self.p_dim = self.obs_dim + self.theta_dim + self.UNC_dim + self.PRICE_dim

        # cost function
        self.stage_cost = self.stage_cost_fn()  # return a function
        self.terminal_cost = self.terminal_cost_fn()

        self.lbg_vcsd = None
        self.ubg_vcsd = None
        self.vsolver = None
        self.dPi = None
        self.dLagV = None

        # Optimization formulation with sensitivity (symbolic)
        self.opt_formulation()

    def opt_formulation(self):
        # Optimization cost and associated constraints
        J = 0
        g = []  # Equality constraints
        hx = []  # Box constraints on states
        hu = []  # Box constraints on inputs
        hsg = []  # Box constraints on sigma

        # input inequalities
        # 0 < p_hp + theta_hp < 3 + sigma_hp
        hu.append(0 - (self.U[4, 0] + self.theta_hp))
        hu.append((self.U[4, 0] + self.theta_hp) - 3)
        # 0.2 + sigma_xv < xv + theta_xv < 0.8 + sigma_xv
        hu.append(0.2 - (self.U[5, 0] + self.theta_xv))
        hu.append((self.U[5, 0] + self.theta_xv) - 0.8)
        # 0 < p_ch + theta_ch < 1 + sigma_ch
        hu.append(0 - (self.U[0, 0] + self.theta_ch_dis[0]))
        hu.append((self.U[0, 0] + self.theta_ch_dis[0]) - 1)
        # 0 < p_dis + theta_dis < 1 + sigma_dis
        hu.append(0 - (self.U[1, 0] + self.theta_ch_dis[1]))
        hu.append((self.U[1, 0] + self.theta_ch_dis[1]) - 1)
        # 0 < p_buy + theta_buy < 5 + sigma_buy
        hu.append(0 - (self.U[2, 0] + self.theta_buy_sell[0]))
        hu.append((self.U[2, 0] + self.theta_buy_sell[0]) - 5)
        # 0 < p_sell + theta_sell < 5 + sigma_sell
        hu.append(0 - (self.U[3, 0] + self.theta_buy_sell[1]))
        hu.append((self.U[3, 0] + self.theta_buy_sell[1]) - 5)

        # sys equalities: power balance
        g.append((self.UNC[1, 0] + self.U[4, 0] + self.U[0, 0] + self.U[3, 0]) -
                 (self.U[1, 0] + self.U[2, 0] + (self.beta * self.A_pv * self.UNC[0, 0])))

        # initial model
        xn = self.env.discrete_model_mpc(self.x, self.U[:, 0], self.UNC[:, 0], self.theta_model)

        for i in range(self.N - 1):
            J += self.gamma ** i * (self.stage_cost(self.X[:, i], self.U[:, i], self.theta,
                                                    self.UNC[:, i], self.PRICE[:, i]))

            ### extra terms in J
            # penalty for big delta u
            # J += 10 * csd.dot((self.U[:-2, i+1] - self.U[:-2, i]), (self.U[:-2, i+1] - self.U[:-2, i]))
            # J += 10 * csd.dot((self.U[-2, i+1] - self.U[-2, i]), self.U[-2, i+1] - self.U[-2, i])
            J += 10 * (self.U[-1, i + 1] - self.U[-1, i]) * (self.U[-1, i + 1] - self.U[-1, i])
            # penalty for conflict
            # J += 100000000 * self.U[0, i] * self.U[1, i]  # charge and discharge
            # J += 100000000 * self.U[2, i] * self.U[3, i]  # buy and sell
            # constraint for conflict
            # g.append(self.U[0, i] * self.U[1, i])
            # g.append(self.U[2, i] * self.U[3, i])

            # model equality
            g.append(self.X[:, i] - xn)
            xn = self.env.discrete_model_mpc(self.X[:, i], self.U[:, i + 1], self.UNC[:, i + 1], self.theta_model)

            # sys equalities
            g.append((self.UNC[1, i + 1] + self.U[4, i + 1] + self.U[0, i + 1] + self.U[3, i + 1]) -
                     (self.U[1, i + 1] + self.U[2, i + 1] + (self.beta * self.A_pv * self.UNC[0, i + 1])))

            # sys inequalities
            # 20 + sigma_t_1,2,3 < t_1,2,3 + theta_t_1,2,3 < 60 + sigma_t_1,2,3
            hx.append(20 - (self.X[4, i] + self.theta_t[0]))
            hx.append((self.X[4, i] + self.theta_t[0]) - 60)
            hx.append(20 - (self.X[5, i] + self.theta_t[1]))
            hx.append((self.X[5, i] + self.theta_t[1]) - 60)
            hx.append(20 - (self.X[6, i] + self.theta_t[2]))
            hx.append((self.X[6, i] + self.theta_t[2]) - 60)
            # 1 + sigma_e < e + theta_e< 4 + sigma_e
            hx.append(1 - (self.X[7, i] + self.theta_e))
            hx.append((self.X[7, i] + self.theta_e) - 4)

            # input inequalities
            # 0 < p_hp + theta_hp < 3 + sigma_hp
            hu.append(0 - (self.U[4, i + 1] + self.theta_hp))
            hu.append((self.U[4, i + 1] + self.theta_hp) - 3)
            # 0.2 + sigma_xv < xv + theta_xv < 0.8 + sigma_xv
            hu.append(0.2 - (self.U[5, i + 1] + self.theta_xv))
            hu.append((self.U[5, i + 1] + self.theta_xv) - 0.8)
            # 0 < p_ch + theta_ch < 1 + sigma_ch
            hu.append(0 - (self.U[0, i + 1] + self.theta_ch_dis[0]))
            hu.append((self.U[0, i + 1] + self.theta_ch_dis[0]) - 1)
            # 0 < p_dis + theta_dis < 1 + sigma_dis
            hu.append(0 - (self.U[1, i + 1] + self.theta_ch_dis[1]))
            hu.append((self.U[1, i + 1] + self.theta_ch_dis[1]) - 1)
            # 0 < p_buy + theta_buy < 5 + sigma_buy
            hu.append(0 - (self.U[2, i + 1] + self.theta_buy_sell[0]))
            hu.append((self.U[2, i + 1] + self.theta_buy_sell[0]) - 5)
            # 0 < p_sell + theta_sell < 5 + sigma_sell
            hu.append(0 - (self.U[3, i + 1] + self.theta_buy_sell[1]))
            hu.append((self.U[3, i + 1] + self.theta_buy_sell[1]) - 5)

        J += self.gamma ** (self.N - 1) * (self.terminal_cost(self.X[:, self.N - 1], self.theta))

        g.append(self.X[:, self.N - 1] - xn)
        hx.append(20 - (self.X[4, self.N - 1] + self.theta_t[0]))
        hx.append((self.X[4, self.N - 1] + self.theta_t[0]) - 60)
        hx.append(20 - (self.X[5, self.N - 1] + self.theta_t[1]))
        hx.append((self.X[5, self.N - 1] + self.theta_t[1]) - 60)
        hx.append(20 - (self.X[6, self.N - 1] + self.theta_t[2]))
        hx.append((self.X[6, self.N - 1] + self.theta_t[2]) - 60)
        hx.append(1 - (self.X[7, self.N - 1] + self.theta_e))
        hx.append((self.X[7, self.N - 1] + self.theta_e) - 4)

        # Constraints
        G = csd.vertcat(*g)
        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx)
        G_vcsd = csd.vertcat(*g, *hu, *hx)

        lbg = [0] * G.shape[0] + [-math.inf] * Hu.shape[0] + [-math.inf] * Hx.shape[0]
        ubg = [0] * G.shape[0] + [0] * Hu.shape[0] + [0] * Hx.shape[0]
        self.lbg_vcsd = csd.vertcat(*lbg)
        self.ubg_vcsd = csd.vertcat(*ubg)

        # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 300,
            "ipopt.print_level": 5,
            "print_time": 1,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-7,
        }
        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": self.P,
            "g": G_vcsd,
        }
        self.vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        # self.dPi, self.dLagV = self.build_sensitivity(J, G, Hu, Hx)
        # self.dR_sensfunc = self.build_sensitivity(J, G, Hu, Hx)

    def stage_cost_fn(self):
        l_spo = self.price_buy * self.u[2] - self.price_sell * self.u[3]
        # cop = self.theta_cop ** 2 * (4 - 0.088 * self.t_out - 0.079 * (0.5 * (self.x[5] + self.x[6])) + 7.253)
        # l_spo += cop
        l_tem = self.env.c_low * (self.x[1] - 23) * (self.x[1] - 27) + self.env.c_low * 4
        # l_extra = 10000 * (self.u[2] + self.u[3]) * (self.u[4] + self.u[5])
        # stage_cost = l_tem + l_spo + l_extra
        stage_cost = l_tem + l_spo
        stage_cost_fn = csd.Function("stage_cost_fn", [self.x, self.u, self.theta, self.unc, self.price], [stage_cost])
        return stage_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = 300 * (self.x[7] - 3) ** 2
        terminal_cost_fn = csd.Function("stage_cost_fn", [self.x, self.theta], [terminal_cost])
        return terminal_cost_fn


# Instantiate the nlp solver
class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug
        self.p_val = np.zeros((self.p_dim, 1))

        self.actor_wt = np.array(cost_params["cost_wt"]) if "cost_wt" in cost_params \
            else np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

        self.X0 = None
        self.soln = None
        self.info = None

    # execute vsolver (solve the nlp problem)
    # return act, self.info
    def act_forward(self, state, act_wt=None, mode="train"):
        act_wt = act_wt if act_wt is not None else self.actor_wt

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.UNC_dim, :] = \
            np.reshape(self.env.uncertainty[:, self.env.t:self.env.t + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.UNC_dim:, :] = \
            np.reshape(self.env.price[:, self.env.t:self.env.t + self.N], (-1, 1), order='F')
        # order='F' reshape the matrix by column

        self.X0 = np.zeros((self.obs_dim + self.action_dim) * self.N)

        self.soln = self.vsolver(
            x0=self.X0,
            p=self.p_val,
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd, )
        fl = self.vsolver.stats()
        if not fl["success"]:
            RuntimeError("Problem is Infeasible")

        opt_var = self.soln["x"].full()
        act = np.array(opt_var[: self.action_dim])[:, 0]

        # add time info as additional infos
        self.info = {"soln": self.soln, "time": self.env.t}

        if self.debug:
            print("Soln")
            print(opt_var[: self.action_dim * self.N, :][:self.action_dim, :].T)
            # print(opt_var[self.action_dim * self.N: self.action_dim * self.N + self.obs_dim * self.N, :]
            #       [:self.obs_dim, :].T)
        return act, self.info


#################
if __name__ == "__main__":
    SafeRL_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    params_path = os.path.join(SafeRL_path, 'Settings/other/smarthome_mpc.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
        print(f'params = {params})')

    ### Environment
    env = SmartHome(params["env_params"])

    ### Agent
    agent = Smart_Home_MPCAgent(env, params["agent_params"])

    ### run MPC
    states, actions, l_spos, l_tems, rewards, rollout_returns = rollout_sample(env, agent, params["n_steps"],
                                                                               mode="mpc")
    data_set = np.concatenate((states, actions, l_spos, l_tems, rewards, rollout_returns))  # stacked vertically

    ### save data
    # save_path = os.path.join(SafeRL_path, 'Project/Results/SmartHome/result_mpc_noise.csv')
    save_path = os.path.join(SafeRL_path, 'Project/Results/SmartHome/result_mpc_test.csv')
    save_data(data_set, save_path)
    print('Simulation finished, results have been saved.')
