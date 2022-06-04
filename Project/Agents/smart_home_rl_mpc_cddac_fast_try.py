import numpy as np
import casadi as csd
import math
from .abstract_agent import TrainableController
from replay_buffer import BasicBuffer
from helpers import tqdm_context


class Smart_Home_MPCAgent(TrainableController):
    def __init__(self, env, agent_params):
        super(Smart_Home_MPCAgent, self).__init__(env=env)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Hyper-parameters
        self._parse_agent_params(**agent_params)

        # Actor initialization
        self.actor = Custom_MPCActor(self.env, self.mpc_horizon, self.cost_params, self.gamma, self.debug)

        ### Critic params initialization
        # dim_sfeature = int(2 * self.obs_dim + 1)
        dim_sfeature = int((self.obs_dim + 2) * (self.obs_dim + 1) / 2)  ### if consider cross terms, use this formula
        # self.vi = 0.01 * np.random.randn(dim_sfeature, 1)
        # self.omega = 0.01 * np.random.randn(self.actor.actor_wt.shape[0], 1)
        # self.nu = 0.01 * np.random.randn(self.actor.actor_wt.shape[0], 1)
        self.vi = np.zeros((dim_sfeature, 1))
        self.omega = np.zeros((self.actor.actor_wt.shape[0], 1))
        self.nu = 0.01 * np.ones((self.actor.actor_wt.shape[0], 1))
        self.adam_m = 0
        self.adam_n = 0

        self.num_policy_update = 0
        self.norm2_delta_theta = []

        self.theta_low_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf,
                                         0, -np.inf, -np.inf,
                                         0, 0,
                                         -np.inf, -np.inf, -np.inf,
                                         -np.inf, -np.inf, -np.inf,
                                         -np.inf, -np.inf, -np.inf, -np.inf
                                         ])

        self.theta_up_bound = np.array([np.inf, np.inf, np.inf, np.inf,
                                        np.inf, np.inf, np.inf,
                                        np.inf, np.inf,
                                        np.inf, np.inf, np.inf,
                                        0.06, 0.1, 0.8,
                                        0.02, 0.02, 0.1, 0.1
                                        ])

        ### Render prep
        self.fig = None
        self.ax = None

    def _parse_agent_params(self, cost_params, eps, gamma, actor_lr, nu_lr, vi_lr, omega_lr, policy_delay,
                            mpc_horizon, debug, train_params):
        self.cost_params = cost_params
        self.eps = eps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.nu_lr = nu_lr
        self.vi_lr = vi_lr
        self.omega_lr = omega_lr
        self.policy_delay = policy_delay
        self.mpc_horizon = mpc_horizon
        self.debug = debug
        self.iterations = train_params["iterations"]
        self.batch_size = train_params["batch_size"]

    def state_to_feature(self, state):
        SS = np.triu(np.outer(state, state))
        size = state.shape[0]
        phi_s = []
        for row in range(size):
            for col in range(row, size):
                phi_s.append(SS[row][col])
        phi_s = np.concatenate((phi_s, state, 1.0), axis=None)[:, None]
        return phi_s

    def get_V_value(self, phi_s):
        V = np.matmul(phi_s.T, self.vi)
        return V

    def state_action_to_feature(self, action, pi, dpi_dtheta):
        phi_sa = np.matmul(dpi_dtheta, (action - pi))
        return phi_sa

    def get_Q_value(self, phi_sa, V):
        Q = np.matmul(phi_sa.T, self.omega) + V
        return Q

    def get_action(self, state, act_wt=None, time=None, mode=None):
        pi, info = self.actor.act_forward(state, act_wt=act_wt, time=time, mode=mode)  # act_wt = self.actor.actor_wt
        if mode == "train":
            info['pi'] = pi

            # ## act = pi + self.eps * (-0.5 + np.random.rand(self.action_dim)) * [1, 1, 5, 5, 3, 1]  # self.eps=0.2
            # act = pi + self.eps * np.random.randn(self.action_dim) * [1, 1, 5, 5, 3, 1]

            # ##gradually decrease the action exploration
            act = pi + (self.eps * np.random.randn(self.action_dim) * [1, 1, 5, 5, 3, 1]) * \
                  (0.9985 ** self.num_policy_update)

            act = act.clip(self.env.action_space.low, self.env.action_space.high)
            # calculate features and save those info
            phi_s = self.state_to_feature(state.squeeze())
            dpi_dtheta_s = self.actor.dPidP(state, info['act_wt'], info)
            phi_sa = self.state_action_to_feature(act[:, None], pi[:, None], dpi_dtheta_s)
            info['phi_s'] = phi_s
            info['phi_sa'] = phi_sa
            info["dpi_dtheta_s"] = dpi_dtheta_s
        else:  # mode == "eval" or "mpc"
            act = pi
            act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def train(self, replay_buffer, train_it):
        delta_dpidpi = 0
        for train_i in tqdm_context(range(train_it), desc="Training Iterations"):
            # print(f'batch training iteration {train_i}')
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(self.batch_size)
            delta_nu = 0
            delta_vi = 0
            delta_omega = 0
            for j, s in enumerate(states):
                phi_s = infos[j]["phi_s"]
                phi_sa = infos[j]["phi_sa"]
                phi_ns = infos[j]["phi_ns"]
                dpi_dtheta_s = infos[j]["dpi_dtheta_s"]
                V_s = self.get_V_value(phi_s)
                Q_sa = self.get_Q_value(phi_sa, V_s)
                V_ns = self.get_V_value(phi_ns)

                td_error = rewards[j] + self.gamma * V_ns - Q_sa
                delta_nu += ((td_error - np.matmul(phi_sa.T, self.nu)) * phi_sa) / self.batch_size
                delta_vi += (td_error * phi_s - self.gamma * np.matmul(phi_sa.T, self.nu) * phi_ns) / self.batch_size
                delta_omega += (td_error * phi_sa) / self.batch_size
                # delta_vi += (td_error * phi_s) / self.batch_size
                # delta_omega += (td_error * phi_sa) / self.batch_size
                delta_dpidpi += np.matmul(dpi_dtheta_s, dpi_dtheta_s.T)

            # print(f'update critic')
            self.nu = self.nu + self.nu_lr * delta_nu
            self.vi = self.vi + self.vi_lr * delta_vi
            self.omega = self.omega + self.omega_lr * delta_omega
            # print(f'self.nu = {self.nu.squeeze()}')
            # print(f'td_error = {td_error}')
            # print(f'self.vi = {self.vi.squeeze()}')
            # print(f'self.omega = {self.omega.squeeze()}')

            if (train_i + 1) % self.policy_delay == 0:
                # print(f'update actor')
                dpi_dpidpi_avg = delta_dpidpi / ((train_i + 1) * self.batch_size)
                delta_theta = np.matmul(dpi_dpidpi_avg, self.omega)

                # print ||\delta_theta||_2
                # print(f'delta_theta = {delta_theta.squeeze()}')
                self.norm2_delta_theta.append(np.linalg.norm(delta_theta.squeeze()))

                delta_dpidpi = 0
                self.num_policy_update += 1
                # Adam
                self.adam_m = 0.9 * self.adam_m + (1 - 0.9) * delta_theta
                m_hat = self.adam_m / (1 - 0.9 ** self.num_policy_update)
                self.adam_n = 0.999 * self.adam_n + (1 - 0.999) * delta_theta ** 2
                n_hat = self.adam_n / (1 - 0.999 ** self.num_policy_update)
                new_theta = self.actor.actor_wt - self.actor_lr * (m_hat / (np.sqrt(n_hat) + 10 ** (-8)))
                self.actor.actor_wt = np.minimum(
                    np.maximum(new_theta.squeeze(), self.theta_low_bound), self.theta_up_bound)[:, None]
                # print(f'self.actor.actor_wt = {self.actor.actor_wt.squeeze()}')
        # print('hi')


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
        self.beta = self.env.beta
        self.A_pv = self.env.A_pv

        # Symbolic variables for optimization problem
        self.x = csd.MX.sym("x", self.obs_dim)
        self.u = csd.MX.sym("u", self.action_dim)
        self.sigma_dim = 10  # dimension of sigma is 10 in our case
        self.sigma = csd.MX.sym("sigma", self.sigma_dim)
        self.X = csd.MX.sym("X", self.obs_dim, self.N)
        self.U = csd.MX.sym("U", self.action_dim, self.N)
        self.Sigma = csd.MX.sym("Sigma", self.sigma_dim, self.N)
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1),
            csd.reshape(self.X, -1, 1),
            csd.reshape(self.Sigma, -1, 1), )

        # Symbolic variables for all parameters: uncertainty, price, theta
        # uncertainty
        # dimension = 3 * N
        self.p_rad = csd.MX.sym('p_rad', 1)
        self.p_app = csd.MX.sym('p_app', 1)
        self.t_out = csd.MX.sym('p_out', 1)
        self.unc = csd.vertcat(self.p_rad, self.p_app, self.t_out)
        self.UNC = csd.MX.sym("UNC", self.unc.size()[0], self.N)
        self.UNC_dim = self.unc.size()[0] * self.N
        # price
        # dimension = 2 * N
        self.Price = csd.MX.sym("Price", 2, 1)
        self.PRICE = csd.MX.sym("PRICE", 2, self.N)
        self.PRICE_dim = 2 * self.N
        # theta
        # dimension = 19
        self.theta_model = csd.MX.sym("theta_model", 4)
        self.theta_in = csd.MX.sym("theta_in", 3)
        self.theta_en = csd.MX.sym("theta_en", 2)
        self.theta_t = csd.MX.sym("theta_t", 3)
        self.theta_hp = csd.MX.sym("theta_hp", 1)
        self.theta_xv = csd.MX.sym("theta_xv", 1)
        self.theta_e = csd.MX.sym("theta_e", 1)
        self.theta_ch_dis = csd.MX.sym("theta_ch_dis", 2)
        self.theta_buy_sell = csd.MX.sym("theta_buy_sell", 2)
        self.theta = csd.vertcat(self.theta_model, self.theta_in, self.theta_en, self.theta_t,
                                 self.theta_hp, self.theta_xv, self.theta_e, self.theta_ch_dis, self.theta_buy_sell)
        self.theta_dim = self.theta.size()[0]

        # [Initial state=8, theta params=21, uncertainty=3*N, price=2*N]
        self.P = csd.vertcat(self.x, self.theta, csd.reshape(self.UNC, -1, 1), csd.reshape(self.PRICE, -1, 1))
        # note that csd.reshape() is reshaped by column
        self.p_dim = self.obs_dim + self.theta_dim + self.UNC_dim + self.PRICE_dim

        # cost function
        self.stage_cost = self.stage_cost_fn()
        self.terminal_cost = self.terminal_cost_fn()

        self.lbg_vcsd = None
        self.ubg_vcsd = None
        self.vsolver = None
        self.dPi = None
        self.dLagV = None

        self.opt_formulation()

    def opt_formulation(self):
        J = 0
        W = np.ones((1, self.sigma_dim)) * [1, 1, 1, 10, 100, 1, 10, 10, 10, 10]
        g = []
        hx = []
        hu = []
        hsg = []

        # input inequalities
        # 0 < p_hp + theta_hp < 3 + sigma_hp
        hu.append(0 - self.U[4, 0])
        hu.append((self.U[4, 0] - self.theta_hp) - (3 + self.Sigma[3, 0]))
        # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
        hu.append((0.2 - self.Sigma[4, 0]) - (self.U[5, 0] + self.theta_xv))
        hu.append((self.U[5, 0] - self.theta_xv) - (0.8 + self.Sigma[4, 0]))
        # 0 < p_ch + theta_ch < 1 + sigma_ch
        hu.append(0 - (self.U[0, 0]))
        hu.append((self.U[0, 0] - self.theta_ch_dis[0]) - (1 + self.Sigma[6, 0]))
        # 0 < p_dis + theta_dis < 1 + sigma_dis
        hu.append(0 - (self.U[1, 0]))
        hu.append((self.U[1, 0] - self.theta_ch_dis[1]) - (1 + self.Sigma[7, 0]))
        # 0 < p_buy + theta_buy < 5 + sigma_buy
        hu.append(0 - (self.U[2, 0]))
        hu.append((self.U[2, 0] - self.theta_buy_sell[0]) - (5 + self.Sigma[8, 0]))
        # 0 < p_sell + theta_sell < 5 + sigma_sell
        hu.append(0 - (self.U[3, 0]))
        hu.append((self.U[3, 0] - self.theta_buy_sell[1]) - (5 + self.Sigma[9, 0]))

        # sys equalities: power balance
        g.append((self.UNC[1, 0] + self.U[4, 0] + self.U[0, 0] + self.U[3, 0]) -
                 (self.U[1, 0] + self.U[2, 0] + (self.beta * self.A_pv * self.UNC[0, 0])))

        # initial model
        xn = self.env.discrete_model_mpc(self.x, self.U[:, 0], self.UNC[:, 0], self.theta_model)

        for i in range(self.N - 1):
            J += self.gamma ** i * (self.stage_cost(self.X[:, i], self.U[:, i], self.theta,
                                                    self.PRICE[:, i]) + W @ self.Sigma[:, i])

            # model equality
            g.append(self.X[:, i] - xn)
            xn = self.env.discrete_model_mpc(self.X[:, i], self.U[:, i + 1], self.UNC[:, i + 1], self.theta_model)

            # sys equalities
            g.append((self.UNC[1, i + 1] + self.U[4, i + 1] + self.U[0, i + 1] + self.U[3, i + 1]) -
                     (self.U[1, i + 1] + self.U[2, i + 1] + (self.beta * self.A_pv * self.UNC[0, i + 1])))

            # sys inequalities
            # 20 + sigma_t_1,2,3 < t_1,2,3 + theta_t_1,2,3 < 60 + sigma_t_1,2,3
            hx.append((20 - self.Sigma[0, i]) - (self.X[4, i] + self.theta_t[0]))
            hx.append((self.X[4, i] - self.theta_t[0]) - (60 + self.Sigma[0, i]))
            hx.append((20 - self.Sigma[1, i]) - (self.X[5, i] + self.theta_t[1]))
            hx.append((self.X[5, i] - self.theta_t[1]) - (60 + self.Sigma[1, i]))
            hx.append((20 - self.Sigma[2, i]) - (self.X[6, i] + self.theta_t[2]))
            hx.append((self.X[6, i] - self.theta_t[2]) - (60 + self.Sigma[2, i]))
            # 1 + sigma_e < e + theta_e< 4 + sigma_e
            hx.append((1 - self.Sigma[5, i]) - (self.X[7, i] + self.theta_e))
            hx.append((self.X[7, i] - self.theta_e) - (4 + self.Sigma[5, i]))

            # input inequalities
            # 0 < p_hp + theta_hp < 3 + sigma_hp
            hu.append(0 - (self.U[4, i + 1]))
            hu.append((self.U[4, i + 1] - self.theta_hp) - (3 + self.Sigma[3, i + 1]))
            # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
            hu.append((0.2 - self.Sigma[4, i + 1]) - (self.U[5, i + 1] + self.theta_xv))
            hu.append((self.U[5, i + 1] - self.theta_xv) - (0.8 + self.Sigma[4, i + 1]))
            # 0 < p_ch + theta_ch < 1 + sigma_ch
            hu.append(0 - (self.U[0, i + 1]))
            hu.append((self.U[0, i + 1] - self.theta_ch_dis[0]) - (1 + self.Sigma[6, i + 1]))
            # 0 < p_dis + theta_dis < 1 + sigma_dis
            hu.append(0 - (self.U[1, i + 1]))
            hu.append((self.U[1, i + 1] - self.theta_ch_dis[1]) - (1 + self.Sigma[7, i + 1]))
            # 0 < p_buy + theta_buy < 5 + sigma_buy
            hu.append(0 - (self.U[2, i + 1]))
            hu.append((self.U[2, i + 1] - self.theta_buy_sell[0]) - (5 + self.Sigma[8, i + 1]))
            # 0 < p_sell + theta_sell < 5 + sigma_sell
            hu.append(0 - (self.U[3, i + 1]))
            hu.append((self.U[3, i + 1] - self.theta_buy_sell[1]) - (5 + self.Sigma[9, i + 1]))

            # slack inequalities
            for _ in range(self.sigma_dim):
                hsg.append(-self.Sigma[_, i])

        J += self.gamma ** (self.N - 1) * (
                self.terminal_cost(self.X[:, self.N - 1], self.theta) + W @ self.Sigma[:, self.N - 1])

        g.append(self.X[:, self.N - 1] - xn)
        hx.append((20 - self.Sigma[0, self.N - 1]) - (self.X[4, self.N - 1] + self.theta_t[0]))
        hx.append((self.X[4, self.N - 1] - self.theta_t[0]) - (60 + self.Sigma[0, self.N - 1]))
        hx.append((20 - self.Sigma[1, self.N - 1]) - (self.X[5, self.N - 1] + self.theta_t[1]))
        hx.append((self.X[5, self.N - 1] - self.theta_t[1]) - (60 + self.Sigma[1, self.N - 1]))
        hx.append((20 - self.Sigma[2, self.N - 1]) - (self.X[6, self.N - 1] + self.theta_t[2]))
        hx.append((self.X[6, self.N - 1] - self.theta_t[2]) - (60 + self.Sigma[2, self.N - 1]))
        hx.append((1 - self.Sigma[5, self.N - 1]) - (self.X[7, self.N - 1] + self.theta_e))
        hx.append((self.X[7, self.N - 1] - self.theta_e) - (4 + self.Sigma[5, self.N - 1]))
        for _ in range(self.sigma_dim):
            hsg.append(-self.Sigma[_, self.N - 1])

        # Constraints
        G = csd.vertcat(*g)
        H = csd.vertcat(*hu, *hx, *hsg)
        GH_vcsd = csd.vertcat(G, H)

        lbg = [0] * G.shape[0] + [-math.inf] * H.shape[0]
        ubg = [0] * G.shape[0] + [0] * H.shape[0]
        self.lbg_vcsd = csd.vertcat(*lbg)
        self.ubg_vcsd = csd.vertcat(*ubg)

        # NLP Problem for value function and policy approximation
        opts_setting = {
            "ipopt.max_iter": 300,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-7,
        }
        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": self.P,
            "g": GH_vcsd,
        }
        self.vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        # self.dPi, self.dLagV = self.build_sensitivity(J, G, Hu, Hx)
        self.dR_sensfunc = self.build_sensitivity(J, G, H)

    def build_sensitivity(self, J, g, h):
        lamb = csd.MX.sym("lamb", g.shape[0])
        mu = csd.MX.sym("mu", h.shape[0])
        mult = csd.vertcat(lamb, mu)

        Lag = J + csd.transpose(lamb) @ g + csd.transpose(mu) @ h

        Lagfunc = csd.Function("Lag", [self.Opt_Vars, mult, self.P], [Lag])
        dLagfunc = Lagfunc.factory("dLagfunc", ["i0", "i1", "i2"], ["jac:o0:i0"])
        dLdw = dLagfunc(self.Opt_Vars, mult, self.P)
        Rr = csd.vertcat(csd.transpose(dLdw), g, mu * h + self.etau)
        z = csd.vertcat(self.Opt_Vars, mult)
        R_kkt = csd.Function("R_kkt", [z, self.P], [Rr])
        dR_sensfunc = R_kkt.factory("dR", ["i0", "i1"], ["jac:o0:i0", "jac:o0:i1"])
        return dR_sensfunc

    def stage_cost_fn(self):
        l_spo = self.Price[0] * self.u[2] - self.Price[1] * self.u[3]
        l_tem = self.theta_in[0] * (self.x[1] - 23 - self.theta_in[1]) * (self.x[1] - 27 - self.theta_in[2])
        stage_cost = l_tem + l_spo
        # stage_cost = l_tem + l_spo
        stage_cost_fn = csd.Function("stage_cost_fn", [self.x, self.u, self.theta, self.Price], [stage_cost])
        return stage_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = self.theta_en[0] * (self.x[7] - self.theta_en[1]) ** 2
        terminal_cost_fn = csd.Function("stage_cost_fn", [self.x, self.theta], [terminal_cost])
        return terminal_cost_fn


class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug

        self.p_val = np.zeros((self.p_dim, 1))

        self.actor_wt = np.concatenate((
            np.zeros(4),
            np.array([5, 0, 0]),
            np.array([300, 3]),
            np.zeros(10)), axis=None)[:, None]

        self.X0 = None
        self.soln = None
        self.info = None

    def act_forward(self, state, act_wt=None, time=None, mode="train"):
        act_wt = act_wt if act_wt is not None else self.actor_wt
        time = time if time is not None else self.env.t

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.UNC_dim, :] = \
            np.reshape(self.env.uncertainty[:, time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.UNC_dim:, :] = \
            np.reshape(self.env.price[:, time:time + self.N], (-1, 1), order='F')
        # order='F' reshape the matrix by column

        self.X0 = 0.01 * np.ones((self.obs_dim + self.action_dim + self.sigma_dim) * self.N)

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
        self.info = {"soln": self.soln, "time": time, "act_wt": act_wt}

        if self.debug:
            print("Soln")
            print(opt_var[: self.action_dim * self.N, :].T)
            print(opt_var[self.action_dim * self.N: self.action_dim * self.N + self.obs_dim * self.N, :].T)
            print(opt_var[self.action_dim * self.N + self.obs_dim * self.N:, :].T)
        return act, self.info

    def dPidP(self, state, act_wt, info):
        soln = info["soln"]
        x = soln["x"].full()
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        # recall the time when calculate info
        time = info["time"]
        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.UNC_dim, :] = \
            np.reshape(self.env.uncertainty[:, time:time + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.UNC_dim:, :] = \
            np.reshape(self.env.price[:, time:time + self.N], (-1, 1), order='F')

        [dRdz, dRdP] = self.dR_sensfunc(z, self.p_val)
        dzdP = (-np.linalg.solve(dRdz, dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim])).T
        # dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim])
        dpi = dzdP[:, :self.action_dim]
        return dpi

    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt
