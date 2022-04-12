import numpy as np
import casadi as csd
import math
from .abstract_agent import TrainableController
from replay_buffer import BasicBuffer
from helpers import tqdm_context


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

        # Critic params
        self.n_sfeature = int(2 * self.obs_dim + 1)
        self.critic_wt = 0.01 * np.random.rand(self.n_sfeature, 1)
        self.adv_wt = 0.01 * np.random.rand(self.actor.actor_wt.shape[0], 1)

        # Render prep
        self.fig = None
        self.ax = None

    # e.g., s1^2+s2^2+s1+s2+1
    # return feature \phi(s)
    def state_to_feature(self, state):
        S = np.diag(np.outer(state, state))
        S = np.concatenate((S, state, np.array([1.0])))
        return S

    # return state value function V(s)
    def get_value(self, state):
        S = self.state_to_feature(state)
        V = S.dot(self.critic_wt)[0]
        return V

    # return action value function Q(s,a)
    # actually not in use
    # def get_Q_value(self, state, act):
    #     S = self.state_to_feature(state)
    #     pi_act, soln = self.get_action(state, self.actor.actor_wt, mode="update")
    #     jacob_pi_act = self.actor.dPidP(state, self.actor.actor_wt, soln)
    #     psi = np.matmul(jacob_pi_act.T, (act - pi_act)[:, None])
    #     Q = np.matmul(psi.T, self.adv_wt)[0] + S.dot(self.critic_wt)[0]
    #     return Q

    # return act, info
    def get_action(self, state, act_wt=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0
        act, info = self.actor.act_forward(state, act_wt=act_wt, mode=mode)
        # act_wt = self.actor.actor_wt
        # with learning, self.actor.actor_wt is updating, so in the next iteration, when sampling,
        #       execute get_action(), i.e., solve the nlp, since MPC parameters theta has changed,
        #       the action will be different.
        act += eps * (np.random.rand(self.action_dim))  # the added noise should de changed
        act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def train(self, replay_buffer, train_it):
        # Critic param update
        Av = np.zeros(shape=(self.critic_wt.shape[0], self.critic_wt.shape[0]))
        bv = np.zeros(shape=(self.critic_wt.shape[0], 1))
        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(
                self.batch_size)

            for j, s in enumerate(states):
                S = self.state_to_feature(s)[:, None]
                temp = (
                        S
                        - (1 - dones[j])
                        * self.gamma
                        * self.state_to_feature(next_states[j])[:, None]
                )
                Av += np.matmul(S, temp.T)
                bv += rewards[j] * S
        # update self.critic_wt
        self.critic_wt = np.linalg.solve(Av, bv)

        # Advantage fn param update
        Aq = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))
        bq = np.zeros(shape=(self.adv_wt.shape[0], 1))
        G = np.zeros(shape=(self.adv_wt.shape[0], self.adv_wt.shape[0]))
        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(
                self.batch_size)

            for j, s in enumerate(states):
                if self.experience_replay:
                    self.env.t = infos[j]["time"]
                    pi_act, info = self.get_action(
                        s, self.actor.actor_wt, mode="update"
                    )
                else:
                    info = infos[j]
                    soln = info["soln"]
                    pi_act = soln["x"].full()[: self.action_dim][:, 0]
                jacob_pi = self.actor.dPidP(
                    s, self.actor.actor_wt, info
                )  # jacob:[n_a, n_sfeature*n_a]
                psi = np.matmul(
                    jacob_pi.T, (actions[j] - pi_act)[:, None]
                )  # psi: [n_sfeature*n_a, 1]

                Aq += np.matmul(psi, psi.T)
                bq += psi * (
                        rewards[j]
                        + (1 - dones[j]) * self.gamma * self.get_value(next_states[j])
                        - self.get_value(s)
                )
                G += np.matmul(jacob_pi.T, jacob_pi)

        if np.linalg.det(Aq) != 0.0:
            # update self.adv_wt
            self.adv_wt = np.linalg.solve(Aq, bq)

            # Policy param update
            # update self.actor.actor_wt
            if self.constrained_updates:
                self.actor.actor_wt = self.actor.param_update(
                    self.actor_lr / self.batch_size,
                    np.matmul(G, self.adv_wt),
                    self.actor.actor_wt,
                )
            else:
                self.actor.actor_wt -= (self.actor_lr / self.batch_size) * np.matmul(
                    G, self.adv_wt
                )
            print("params updated")

        print(self.actor.actor_wt)

    def _parse_agent_params(
            self,
            cost_params,
            eps,
            gamma,
            actor_lr,
            horizon,
            debug,
            train_params,
            constrained_updates=True,
            experience_replay=False,
    ):
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
        self.etau = 1e-6
        self.th_param = th_param
        self.upper_tri = upper_tri

        # Symbolic variables for optimization problem
        self.x = csd.SX.sym("x", self.obs_dim)
        self.u = csd.SX.sym("u", self.action_dim)
        self.sigma_dim = 10  # dimension of sigma is 10 in our case
        self.sigma = csd.SX.sym("sigma", self.sigma_dim)
        self.X = csd.SX.sym("X", self.obs_dim, self.N)
        self.U = csd.SX.sym("U", self.action_dim, self.N)
        self.Sigma = csd.SX.sym("Sigma", self.sigma_dim, self.N)
        self.Opt_Vars = csd.vertcat(
            csd.reshape(self.U, -1, 1),
            csd.reshape(self.X, -1, 1),
            csd.reshape(self.Sigma, -1, 1), )

        # Symbolic variables for all parameters: uncertainty, price, theta
        # uncertainty
        # dimension = 3 * N
        self.p_rad = csd.SX.sym('p_rad', 1)
        self.p_app = csd.SX.sym('p_app', 1)
        self.t_out = csd.SX.sym('p_out', 1)
        self.unc = csd.vertcat(self.p_rad, self.p_app, self.t_out)
        self.UNC = csd.SX.sym("UNC", 3, self.N)
        self.UNC_dim = 3 * self.N
        # price
        # dimension = 2 * N
        self.price_buy = csd.SX.sym('price_buy', 1)
        self.price_sell = csd.SX.sym('price_sell', 1)
        self.price = csd.vertcat(self.price_buy, self.price_sell)
        self.PRICE = csd.SX.sym("PRICE", 2, self.N)
        self.PRICE_dim = 2 * self.N
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
        self.theta_dim = 20

        # [Initial state=8, theta params=20, uncertainty=3*N, price=2*N]
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

        # Optimization formulation with sensitivity (symbolic)
        self.opt_formulation()

    def opt_formulation(self):
        # Optimization cost and associated constraints
        J = 0
        W = np.ones((1, self.sigma_dim))
        g = []  # Equality constraints
        hx = []  # Box constraints on states
        hu = []  # Box constraints on inputs
        hsg = []  # Box constraints on sigma

        # input inequalities
        # 0 < p_hp + theta_hp < 3 + sigma_hp
        hu.append(0 - (self.U[7, 0] + self.theta_hp))
        hu.append((self.U[7, 0] + self.theta_hp) - (3 + self.Sigma[3, 0]))
        # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
        hu.append((0.2 + self.Sigma[4, 0]) - (self.U[8, 0] + self.theta_xv))
        hu.append((self.U[8, 0] + self.theta_xv) - (0.8 + self.Sigma[4, 0]))
        # 0 < p_ch + theta_ch < 1 + sigma_ch
        hu.append(0 - (self.U[2, 0] + self.U[3, 0] + self.theta_ch_dis[0]))
        hu.append((self.U[2, 0] + self.U[3, 0] + self.theta_ch_dis[0]) - (1 + self.Sigma[6, 0]))
        # 0 < p_dis + theta_dis < 1 + sigma_dis
        hu.append(0 - (self.U[4, 0] + self.U[5, 0] + self.theta_ch_dis[1]))
        hu.append((self.U[4, 0] + self.U[5, 0] + self.theta_ch_dis[1]) - (1 + self.Sigma[7, 0]))
        # 0 < p_buy + theta_buy < 5 + sigma_buy
        hu.append(0 - (self.U[3, 0] + self.U[6, 0] + self.theta_buy_sell[0]))
        hu.append((self.U[3, 0] + self.U[6, 0] + self.theta_buy_sell[0]) - (5 + self.Sigma[8, 0]))
        # 0 < p_sell + theta_sell < 5 + sigma_sell
        hu.append(0 - (self.U[0, 0] + self.U[4, 0] + self.theta_buy_sell[1]))
        hu.append((self.U[0, 0] + self.U[4, 0] + self.theta_buy_sell[1]) - (5 + self.Sigma[9, 0]))

        # sys equalities: power balance
        beta = 0.429
        A_pv = 35
        g.append((beta * A_pv * self.UNC[0, 0]) - (self.U[0, 0] + self.U[1, 0] + self.U[2, 0]))
        g.append((self.UNC[1, 0] + self.U[7, 0]) - (self.U[6, 0] + self.U[1, 0] + self.U[5, 0]))

        # initial model
        xn = self.env.discrete_model_mpc(self.x, self.U[:, 0], self.UNC[:, 0], self.theta_model)

        for i in range(self.N - 1):
            J += self.gamma ** i * (self.stage_cost(self.X[:, i], self.U[:, i], self.theta,
                                                    self.UNC[:, i], self.PRICE[:, i]) + W @ self.Sigma[:, i])

            # model equality
            g.append(self.X[:, i] - xn)
            xn = self.env.discrete_model_mpc(self.X[:, i], self.U[:, i + 1], self.UNC[:, i + 1], self.theta_model)

            # sys equalities
            g.append((beta * A_pv * self.UNC[0, i + 1]) -
                     (self.U[0, i + 1] + self.U[1, i + 1] + self.U[2, i + 1]))
            g.append((self.UNC[1, i + 1] + self.U[7, i + 1]) -
                     (self.U[6, i + 1] + self.U[1, i + 1] + self.U[5, i + 1]))

            # sys inequalities
            # 20 + sigma_t_1,2,3 < t_1,2,3 + theta_t_1,2,3 < 60 + sigma_t_1,2,3
            hx.append((20 + self.Sigma[0, i]) - (self.X[4, i] + self.theta_t[0]))
            hx.append((self.X[4, i] + self.theta_t[0]) - (60 + self.Sigma[0, i]))
            hx.append((20 + self.Sigma[1, i]) - (self.X[5, i] + self.theta_t[1]))
            hx.append((self.X[5, i] + self.theta_t[1]) - (60 + self.Sigma[1, i]))
            hx.append((20 + self.Sigma[2, i]) - (self.X[6, i] + self.theta_t[2]))
            hx.append((self.X[6, i] + self.theta_t[2]) - (60 + self.Sigma[2, i]))
            # 1 + sigma_e < e + theta_e< 4 + sigma_e
            hx.append((1 + self.Sigma[5, i]) - (self.X[7, i] + self.theta_e))
            hx.append((self.X[7, i] + self.theta_e) - (4 + self.Sigma[5, i]))

            # input inequalities
            # 0 < p_hp + theta_hp < 3 + sigma_hp
            hu.append(0 - (self.U[7, i + 1] + self.theta_hp))
            hu.append((self.U[7, i + 1] + self.theta_hp) - (3 + self.Sigma[3, i + 1]))
            # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
            hu.append((0.2 + self.Sigma[4, i + 1]) - (self.U[8, i + 1] + self.theta_xv))
            hu.append((self.U[8, i + 1] + self.theta_xv) - (0.8 + self.Sigma[4, i + 1]))
            # 0 < p_ch + theta_ch < 1 + sigma_ch
            hu.append(0 - (self.U[2, i + 1] + self.U[3, i + 1] + self.theta_ch_dis[0]))
            hu.append((self.U[2, i + 1] + self.U[3, i + 1] + self.theta_ch_dis[0]) - (1 + self.Sigma[6, i + 1]))
            # 0 < p_dis + theta_dis < 1 + sigma_dis
            hu.append(0 - (self.U[4, i + 1] + self.U[5, i + 1] + self.theta_ch_dis[1]))
            hu.append((self.U[4, i + 1] + self.U[5, i + 1] + self.theta_ch_dis[1]) - (1 + self.Sigma[7, i + 1]))
            # 0 < p_buy + theta_buy < 5 + sigma_buy
            hu.append(0 - (self.U[3, i + 1] + self.U[6, i + 1] + self.theta_buy_sell[0]))
            hu.append((self.U[3, i + 1] + self.U[6, i + 1] + self.theta_buy_sell[0]) - (5 + self.Sigma[8, i + 1]))
            # 0 < p_sell + theta_sell < 5 + sigma_sell
            hu.append(0 - (self.U[0, i + 1] + self.U[4, i + 1] + self.theta_buy_sell[1]))
            hu.append((self.U[0, i + 1] + self.U[4, i + 1] + self.theta_buy_sell[1]) - (5 + self.Sigma[9, i + 1]))

            # slack inequalities
            for _ in range(self.sigma_dim):
                hsg.append(-self.Sigma[_, i])

        J += self.gamma ** (self.N - 1) * (
                self.terminal_cost(self.X[:, self.N - 1], self.theta) + W @ self.Sigma[:, self.N - 1])

        g.append(self.X[:, self.N - 1] - xn)
        hx.append((20 + self.Sigma[0, self.N - 1]) - (self.X[4, self.N - 1] + self.theta_t[0]))
        hx.append((self.X[4, self.N - 1] + self.theta_t[0]) - (60 + self.Sigma[0, self.N - 1]))
        hx.append((20 + self.Sigma[1, self.N - 1]) - (self.X[5, self.N - 1] + self.theta_t[1]))
        hx.append((self.X[5, self.N - 1] + self.theta_t[1]) - (60 + self.Sigma[1, self.N - 1]))
        hx.append((20 + self.Sigma[2, self.N - 1]) - (self.X[6, self.N - 1] + self.theta_t[2]))
        hx.append((self.X[6, self.N - 1] + self.theta_t[2]) - (60 + self.Sigma[2, self.N - 1]))
        hx.append((1 + self.Sigma[5, self.N - 1]) - (self.X[7, self.N - 1] + self.theta_e))
        hx.append((self.X[7, self.N - 1] + self.theta_e) - (4 + self.Sigma[5, self.N - 1]))
        for _ in range(self.sigma_dim):
            hsg.append(-self.Sigma[_, self.N - 1])

        for i in range(self.sigma_dim * self.N):
            hx.append(hsg[i])

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
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.mu_target": self.etau,
            "ipopt.mu_init": self.etau,
            "ipopt.acceptable_tol": 1e-5,
            "ipopt.acceptable_obj_change_tol": 1e-5,
        }
        vnlp_prob = {
            "f": J,
            "x": self.Opt_Vars,
            "p": self.P,
            "g": G_vcsd,
        }
        self.vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        # self.dPi, self.dLagV = self.build_sensitivity(J, G, Hu, Hx)
        self.dR_sensfunc = self.build_sensitivity(J, G, Hu, Hx)

    def build_sensitivity(self, J, G, Hu, Hx):
        # Sensitivity
        lamb = csd.SX.sym("lambda", G.shape[0])
        mu_u = csd.SX.sym("muu", Hu.shape[0])
        mu_x = csd.SX.sym("mux", Hx.shape[0])
        mult = csd.vertcat(lamb, mu_u, mu_x)

        Lag = (J
               + csd.transpose(lamb) @ G
               + csd.transpose(mu_u) @ Hu
               + csd.transpose(mu_x) @ Hx)

        Lagfunc = csd.Function("Lag", [self.Opt_Vars, mult, self.P], [Lag])
        dLagfunc = Lagfunc.factory("dLagfunc", ["i0", "i1", "i2"], ["jac:o0:i0"])
        dLdw = dLagfunc(self.Opt_Vars, mult, self.P)
        Rr = csd.vertcat(csd.transpose(dLdw), G, mu_u * Hu + self.etau, mu_x * Hx + self.etau)
        z = csd.vertcat(self.Opt_Vars, mult)
        R_kkt = csd.Function("R_kkt", [z, self.P], [Rr])
        dR_sensfunc = R_kkt.factory("dR", ["i0", "i1"], ["jac:o0:i0", "jac:o0:i1"])
        # [dRdz, dRdP] = dR_sensfunc(z, self.P)
        # dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim]
        # dPi = csd.Function("dPi", [z, self.P], [dzdP])
        # return dPi, dLagfunc
        return dR_sensfunc

    def stage_cost_fn(self):
        l_spo = self.price_buy * (self.u[3] + self.u[6]) - self.price_sell * (self.u[0] + self.u[4])
        cop = self.theta_cop ** 2 * (4 - 0.088 * self.t_out - 0.079 * (0.5 * (self.x[5] + self.x[6])) + 7.253)
        l_spo += cop
        l_tem = self.theta_in[0] ** 2 * (self.x[1] - 23 - self.theta_in[1]) * (self.x[1] - 27 - self.theta_in[2])
        l_extra = 10000 * (self.u[2] + self.u[3]) * (self.u[4] + self.u[5])
        stage_cost = l_tem + l_spo + l_extra
        stage_cost_fn = csd.Function("stage_cost_fn", [self.x, self.u, self.theta, self.unc, self.price], [stage_cost])
        return stage_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = self.theta_en[0] ** 2 * (self.x[7] - self.theta_en[1]) ** 2
        terminal_cost_fn = csd.Function("stage_cost_fn", [self.x, self.theta], [terminal_cost])
        return terminal_cost_fn


# Instantiate the nlp solver
# as actor has two functions:
#   1. act_forward(state, act_wt)
#   2. dPidP(state, act_wt, slon)
class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug
        self.p_val = np.zeros((self.p_dim, 1))

        # Actor param
        # self.actor_wt = np.array(cost_params["cost_wt"]) if "cost_wt" in cost_params \
        #    else 0.01 * np.random.rand(self.theta_dim, 1)
        self.actor_wt = np.array(cost_params["cost_wt"]) if "cost_wt" in cost_params \
            else 0.01 * np.random.rand(self.theta_dim, 1)

        # Test run
        # _ = self.act_forward(self.env.reset())
        self.X0 = None
        self.soln = None
        self.info =None

    # execute vsolver (solve the nlp problem)
    # return act, self.info
    def act_forward(self, state, act_wt=None, mode="train"):
        act_wt = act_wt if act_wt is not None else self.actor_wt

        self.p_val[: self.obs_dim, 0] = state
        self.p_val[self.obs_dim:self.obs_dim + self.theta_dim, :] = act_wt

        self.p_val[self.obs_dim + self.theta_dim:self.obs_dim + self.theta_dim + self.UNC_dim, :] = \
            np.reshape(self.env.uncertainty[:, self.env.t:self.env.t + self.N], (-1, 1), order='F')
        self.p_val[self.obs_dim + self.theta_dim + self.UNC_dim:, :] =\
            np.reshape(self.env.price[:, self.env.t:self.env.t + self.N], (-1, 1), order='F')
        # order='F' reshape the matrix by column

        self.X0 = np.zeros((self.obs_dim + self.action_dim + self.sigma_dim) * self.N)

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
        dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim]
        jacob_act = dzdP.full()

        return jacob_act[: self.action_dim, :]

    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt
