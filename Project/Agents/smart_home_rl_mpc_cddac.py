import numpy as np
import casadi as csd
import math
from .abstract_agent import TrainableController
from replay_buffer import BasicBuffer
from helpers import tqdm_context


class Smart_Home_MPCAgent(TrainableController):
    def __init__(self, env, agent_params, seed):
        super().__init__(env)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Hyper-parameters
        self._parse_agent_params(**agent_params)
        self.rng1 = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed)

        # Actor initialization
        self.actor = Custom_MPCActor(
            self.env, self.mpc_horizon, self.cost_params, self.gamma, self.debug
        )

        # Critic params initialization
        # dim_sfeature = int(2 * self.obs_dim + 1)
        dim_sfeature = int(
            (self.obs_dim + 2) * (self.obs_dim + 1) / 2
        )  # if consider cross terms, use this formula
        # self.vi = 0.01 * self.rng1.random((dim_sfeature, 1))
        # self.omega = 0.01 * self.rng1.random((self.actor.actor_wt.shape[0], 1))
        # self.nu = 0.01 * self.rng1.random((self.actor.actor_wt.shape[0], 1))
        self.vi = np.zeros((dim_sfeature, 1))
        self.omega = np.zeros((self.actor.actor_wt.shape[0], 1))
        self.nu = 0.01 * np.ones((self.actor.actor_wt.shape[0], 1))
        self.adam_m = 0
        self.adam_n = 0
        self.num_policy_update = 0

        # Theta bounds
        self.theta_low_bound = np.array(
            [
                [
                    0.0,
                    -5.0,
                    -5.0,
                    0.0,
                    0.0,
                    -20.0,
                    -20.0,
                    -20.0,
                    -1.0,
                ]
            ]
        ).T

        self.theta_up_bound = np.array(
            [
                [
                    5.0,
                    5.0,
                    np.inf,
                    4.0,
                    20.0,
                    20.0,
                    20.0,
                    1.0,
                ]
            ]
        ).T

    def state_to_feature(self, state):
        ss = np.triu(np.outer(state, state))
        size = state.shape[0]
        phi_s = []
        for row in range(size):
            for col in range(row, size):
                phi_s.append(ss[row][col])
        phi_s = np.concatenate((phi_s, state, 1.0), axis=None)[:, None]
        return phi_s

    def get_V_value(self, phi_s):
        V = np.matmul(phi_s.T, self.vi)
        return V

    def state_action_to_feature(self, action, pi, dpi_dtheta):
        phi_sa = np.matmul(dpi_dtheta.T, (action - pi))
        return phi_sa

    def get_Q_value(self, phi_sa, V):
        Q = np.matmul(phi_sa.T, self.omega) + V
        return Q

    def get_action(self, state, act_wt=None, time=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0
        act, info = self.actor.act_forward(
            state, act_wt=act_wt, time=time
        )  # act_wt = self.actor.actor_wt
        act += eps * (self.rng2.random(self.action_dim) - 0.5)  # * [1, 1, 5, 5, 3, 1]
        act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def train(self, replay_buffer: BasicBuffer):
        batch_size = min(self.batch_size, replay_buffer.size)
        train_it = min(self.iterations, int(3.0 * replay_buffer.size / batch_size))

        delta_dpidpi = 0
        for train_i in tqdm_context(range(train_it), desc="Training Iterations"):
            (
                states,
                obss,
                actions,
                rewards,
                next_states,
                next_obss,
                dones,
                infos,
            ) = replay_buffer.sample(batch_size)
            delta_nu = 0
            delta_vi = 0
            delta_omega = 0

            for j, s in enumerate(states):
                # all info need for (s,a)
                phi_s = self.state_to_feature(s.squeeze())
                V_s = self.get_V_value(phi_s)
                info_s = infos[j]
                pi_s = info_s["soln"]["x"].full()[: self.action_dim]
                action_s = actions[j][:, None]
                dpi_dtheta_s = self.actor.dPidP(s, info_s["act_wt"], info_s)
                phi_sa = self.state_action_to_feature(action_s, pi_s, dpi_dtheta_s)
                Q_sa = self.get_Q_value(phi_sa, V_s)

                # all info need for (s', pi_s')
                ns = next_states[j]
                phi_ns = self.state_to_feature(ns.squeeze())
                V_ns = self.get_V_value(phi_ns)

                td_error = rewards[j] + self.gamma * V_ns - Q_sa
                delta_nu += (
                    (td_error - np.matmul(phi_sa.T, self.nu)) * phi_sa
                ) / batch_size
                delta_vi += (
                    td_error * phi_s
                    - self.gamma * np.matmul(phi_sa.T, self.nu) * phi_ns
                ) / batch_size
                delta_omega += (td_error * phi_sa) / batch_size
                delta_dpidpi += np.matmul(dpi_dtheta_s.T, dpi_dtheta_s)

            self.nu = self.nu + self.nu_lr * delta_nu
            self.vi = self.vi + self.vi_lr * delta_vi
            self.omega = self.omega + self.omega_lr * delta_omega

            if (train_i + 1) % self.policy_delay == 0:
                # print(f"update actor")
                dpi_dpidpi_avg = delta_dpidpi / ((train_i + 1) * batch_size)
                delta_theta = np.matmul(dpi_dpidpi_avg, self.omega)
                delta_dpidpi = 0
                self.num_policy_update += 1

                # Adam
                self.adam_m = 0.9 * self.adam_m + (1 - 0.9) * delta_theta
                m_hat = self.adam_m / (1 - 0.9**self.num_policy_update)
                self.adam_n = 0.999 * self.adam_n + (1 - 0.999) * delta_theta**2
                n_hat = self.adam_n / (1 - 0.999**self.num_policy_update)
                new_theta = self.actor.actor_wt - self.actor_lr * (
                    m_hat / (np.sqrt(n_hat) + 10 ** (-8))
                )
                # self.actor.actor_wt = new_theta.clip(
                #     self.theta_low_bound, self.theta_up_bound
                # )
                self.actor.actor_wt = new_theta
        print(self.actor.actor_wt.T)
        return {"l_theta": self.actor.actor_wt}

    def _parse_agent_params(
        self,
        cost_params,
        eps,
        gamma,
        actor_lr,
        nu_lr,
        vi_lr,
        omega_lr,
        policy_delay,
        mpc_horizon,
        debug,
        train_params,
    ):
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
        self.sigma_dim = 4  # dimension of sigma is 10 in our case
        self.sigma = csd.MX.sym("sigma", self.sigma_dim)
        self.X = csd.MX.sym("X", self.obs_dim, self.N)
        self.U = csd.MX.sym("U", self.action_dim, self.N)
        self.Sigma = csd.MX.sym("Sigma", self.sigma_dim, self.N)
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1),
            csd.reshape(self.X, -1, 1),
            csd.reshape(self.Sigma, -1, 1),
        )

        # Symbolic variables for all parameters: uncertainty, price, theta
        # uncertainty
        # dimension = 3 * N
        self.p_rad = csd.MX.sym("p_rad", 1)
        self.p_app = csd.MX.sym("p_app", 1)
        self.t_out = csd.MX.sym("p_out", 1)
        self.unc = csd.vertcat(self.p_rad, self.p_app, self.t_out)
        self.UNC = csd.MX.sym("UNC", self.unc.size()[0], self.N)
        self.UNC_dim = self.unc.size()[0] * self.N
        # price
        # dimension = 2 * N
        self.price_buy = csd.MX.sym("price_buy", 1)
        self.price_sell = csd.MX.sym("price_sell", 1)
        self.price = csd.vertcat(self.price_buy, self.price_sell)
        self.PRICE = csd.MX.sym("PRICE", self.price.size()[0], self.N)
        self.PRICE_dim = self.price.size()[0] * self.N
        # theta
        # dimension = 19
        # self.theta_model = csd.MX.sym("theta_model", 4)
        self.theta_in = csd.MX.sym("theta_in", 3)
        self.theta_en = csd.MX.sym("theta_en", 2)
        self.theta_t = csd.MX.sym("theta_t", 3)
        # self.theta_hp = csd.MX.sym("theta_hp", 1)
        # self.theta_xv = csd.MX.sym("theta_xv", 1)
        self.theta_e = csd.MX.sym("theta_e", 1)
        # self.theta_ch_dis = csd.MX.sym("theta_ch_dis", 2)
        # self.theta_buy_sell = csd.MX.sym("theta_buy_sell", 2)
        self.theta = csd.vertcat(
            self.theta_in,
            self.theta_en,
            self.theta_t,
            self.theta_e,
        )
        self.theta_dim = self.theta.size()[0]

        # [Initial state=8, theta params=19, uncertainty=3*N, price=2*N]
        self.P = csd.vertcat(
            self.x,
            self.theta,
            csd.reshape(self.UNC, -1, 1),
            csd.reshape(self.PRICE, -1, 1),
        )
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
        self.dR_sensfunc = None
        self.drz = None

        # Optimization formulation with sensitivity (symbolic)
        self.opt_formulation()

    def opt_formulation(self):
        J = 0
        W = np.ones((1, self.sigma_dim))
        g = []
        hx = []
        hu = []
        hsg = []

        # input inequalities
        # 0 < p_hp + theta_hp < 3 + sigma_hp
        hu.append(0 - self.U[4, 0])
        hu.append(self.U[4, 0] - 3)
        # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
        hu.append(0.2 - self.U[5, 0])
        hu.append(self.U[5, 0] - 0.8)
        # 0 < p_ch + theta_ch < 1 + sigma_ch
        hu.append(0 - self.U[0, 0])
        hu.append(self.U[0, 0] - 1)
        # 0 < p_dis + theta_dis < 1 + sigma_dis
        hu.append(0 - self.U[1, 0])
        hu.append(self.U[1, 0] - 1)
        # 0 < p_buy + theta_buy < 5 + sigma_buy
        hu.append(0 - self.U[2, 0])
        hu.append(self.U[2, 0] - 5)
        # 0 < p_sell + theta_sell < 5 + sigma_sell
        hu.append(0 - self.U[3, 0])
        hu.append(self.U[3, 0] - 5)

        # sys equalities: power balance
        g.append(
            (self.UNC[1, 0] + self.U[4, 0] + self.U[0, 0] + self.U[3, 0])
            - (self.U[1, 0] + self.U[2, 0] + (self.beta * self.A_pv * self.UNC[0, 0]))
        )

        # initial model
        xn = self.env.discrete_model_mpc(
            self.x, self.U[:, 0], self.UNC[:, 0], np.zeros((4, 1))
        )

        for i in range(self.N - 1):
            J += self.gamma**i * (
                self.stage_cost(
                    self.X[:, i], self.U[:, i], self.theta, self.PRICE[:, i]
                )
                + W @ self.Sigma[:, i]
            )

            # model equality
            g.append(self.X[:, i] - xn)
            xn = self.env.discrete_model_mpc(
                self.X[:, i], self.U[:, i + 1], self.UNC[:, i + 1], np.zeros((4, 1))
            )

            # sys equalities
            g.append(
                (
                    self.UNC[1, i + 1]
                    + self.U[4, i + 1]
                    + self.U[0, i + 1]
                    + self.U[3, i + 1]
                )
                - (
                    self.U[1, i + 1]
                    + self.U[2, i + 1]
                    + (self.beta * self.A_pv * self.UNC[0, i + 1])
                )
            )

            # sys inequalities
            # 20 + sigma_t_1,2,3 < t_1,2,3 + theta_t_1,2,3 < 60 + sigma_t_1,2,3
            hx.append((20 + self.Sigma[0, i]) - (self.X[4, i] + self.theta_t[0]))
            hx.append((self.X[4, i] + self.theta_t[0]) - (60 + self.Sigma[0, i]))
            hx.append((20 + self.Sigma[1, i]) - (self.X[5, i] + self.theta_t[1]))
            hx.append((self.X[5, i] + self.theta_t[1]) - (60 + self.Sigma[1, i]))
            hx.append((20 + self.Sigma[2, i]) - (self.X[6, i] + self.theta_t[2]))
            hx.append((self.X[6, i] + self.theta_t[2]) - (60 + self.Sigma[2, i]))
            # 1 + sigma_e < e + theta_e< 4 + sigma_e
            hx.append((1 + self.Sigma[3, i]) - (self.X[7, i] + self.theta_e))
            hx.append((self.X[7, i] + self.theta_e) - (4 + self.Sigma[3, i]))

            # input inequalities
            # 0 < p_hp + theta_hp < 3 + sigma_hp
            hu.append(0 - self.U[4, i + 1])
            hu.append(self.U[4, i + 1] - 3)
            # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
            hu.append(0.2 - self.U[5, i + 1])
            hu.append(self.U[5, i + 1] - 0.8)
            # 0 < p_ch + theta_ch < 1 + sigma_ch
            hu.append(0 - self.U[0, i + 1])
            hu.append(self.U[0, i + 1] - 1)
            # 0 < p_dis + theta_dis < 1 + sigma_dis
            hu.append(0 - self.U[1, i + 1])
            hu.append(self.U[1, i + 1] - 1)
            # 0 < p_buy + theta_buy < 5 + sigma_buy
            hu.append(0 - self.U[2, i + 1])
            hu.append(self.U[2, i + 1] - 5)
            # 0 < p_sell + theta_sell < 5 + sigma_sell
            hu.append(0 - self.U[3, i + 1])
            hu.append(self.U[3, i + 1] - 5)

            # slack inequalities
            for _ in range(self.sigma_dim):
                hsg.append(-self.Sigma[_, i])

        J += self.gamma ** (self.N - 1) * (
            self.terminal_cost(self.X[:, self.N - 1], self.theta)
            + W @ self.Sigma[:, self.N - 1]
        )

        g.append(self.X[:, self.N - 1] - xn)
        hx.append(
            (20 + self.Sigma[0, self.N - 1]) - (self.X[4, self.N - 1] + self.theta_t[0])
        )
        hx.append(
            (self.X[4, self.N - 1] + self.theta_t[0]) - (60 + self.Sigma[0, self.N - 1])
        )
        hx.append(
            (20 + self.Sigma[1, self.N - 1]) - (self.X[5, self.N - 1] + self.theta_t[1])
        )
        hx.append(
            (self.X[5, self.N - 1] + self.theta_t[1]) - (60 + self.Sigma[1, self.N - 1])
        )
        hx.append(
            (20 + self.Sigma[2, self.N - 1]) - (self.X[6, self.N - 1] + self.theta_t[2])
        )
        hx.append(
            (self.X[6, self.N - 1] + self.theta_t[2]) - (60 + self.Sigma[2, self.N - 1])
        )
        hx.append(
            (1 + self.Sigma[3, self.N - 1]) - (self.X[7, self.N - 1] + self.theta_e)
        )
        hx.append(
            (self.X[7, self.N - 1] + self.theta_e) - (4 + self.Sigma[3, self.N - 1])
        )
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
        self.dR_sensfunc, self.dPi, self.drz = self.build_sensitivity(J, G, H)

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

        [dRdz, dRdP] = dR_sensfunc(z, self.P)
        drz = csd.Function("drz", [z, self.P], [dRdz])
        dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim : self.obs_dim + self.theta_dim]
        dPi = csd.Function("dPi", [z, self.P], [dzdP[: self.action_dim, :]])
        return dR_sensfunc, dPi, drz

    def stage_cost_fn(self):
        l_spo = self.price_buy * self.u[2] - self.price_sell * self.u[3]
        l_tem = (
            self.theta_in[0]
            * (self.x[1] - 23 - self.theta_in[1])
            * (self.x[1] - 27 - self.theta_in[2])
        )
        stage_cost = l_tem + l_spo
        stage_cost_fn = csd.Function(
            "stage_cost_fn", [self.x, self.u, self.theta, self.price], [stage_cost]
        )
        return stage_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = self.theta_en[0] * (self.x[7] - self.theta_en[1]) ** 2
        terminal_cost_fn = csd.Function(
            "stage_cost_fn", [self.x, self.theta], [terminal_cost]
        )
        return terminal_cost_fn


class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug

        self.p_val = np.zeros((self.p_dim, 1))

        self.actor_wt = np.concatenate(
            (np.array([5.0, 0.0, 0.0]), np.array([3.0, 3.0]), np.zeros(4)),
            axis=None,
        )[:, None]

        # Test run
        s_test, o_test = self.env.reset()
        _ = self.act_forward(o_test)
        self.X0 = None
        self.soln = None
        self.info = None

    def act_forward(self, state, act_wt=None, time=None):
        act_wt = act_wt if act_wt is not None else self.actor_wt
        time = time if time is not None else self.env.t

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim : self.obs_dim + self.theta_dim, :] = act_wt

        self.p_val[
            self.obs_dim
            + self.theta_dim : self.obs_dim
            + self.theta_dim
            + self.UNC_dim,
            :,
        ] = np.reshape(
            self.env.uncertainty[:, time : time + self.N], (-1, 1), order="F"
        )
        self.p_val[self.obs_dim + self.theta_dim + self.UNC_dim :, :] = np.reshape(
            self.env.price[:, time : time + self.N], (-1, 1), order="F"
        )
        # order='F' reshape the matrix by column

        self.X0 = 0.01 * np.ones(
            (self.obs_dim + self.action_dim + self.sigma_dim) * self.N
        )

        self.soln = self.vsolver(
            x0=self.X0,
            p=self.p_val,
            lbg=self.lbg_vcsd,
            ubg=self.ubg_vcsd,
        )
        fl = self.vsolver.stats()
        if not fl["success"]:
            RuntimeError("Problem is Infeasible")

        opt_var = self.soln["x"].full()
        act = np.array(opt_var[: self.action_dim])[:, 0]

        # add time info as additional infos
        self.info = {"soln": self.soln, "time": time, "act_wt": act_wt, "p": self.p_val}
        return act, self.info

    def dPidP(self, state, act_wt, info):
        soln = info["soln"]
        x = soln["x"].full()
        p_val = info["p"]
        lam_g = soln["lam_g"].full()
        z = np.concatenate((x, lam_g), axis=0)

        self.p_val = p_val
        dpi = self.dPi(z, self.p_val).full()
        return dpi

    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt
