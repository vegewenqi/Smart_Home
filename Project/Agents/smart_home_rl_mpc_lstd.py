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

        # Actor initialization
        self.actor = Custom_MPCActor(self.env, self.h, self.cost_params, self.gamma)

        # Critic params
        self.n_sfeature = int((self.obs_dim+2) * (self.obs_dim + 1)/2)
        self.critic_wt = 0.01 * np.random.rand(self.n_sfeature, 1)
        self.adv_wt = 0.01 * np.random.rand(self.actor.actor_wt.shape[0], 1)

        # Parameter bounds
        self.theta_low_bound = np.array([[-np.inf, -np.inf, -np.inf, -np.inf,
                                         1.0,
                                         0.0, 0.0,
                                         -5.0, -5.0, -5.0,
                                         -0.3, -0.1, -0.4,
                                         -0.1, -0.1, -0.5, -0.5]]).T
        self.theta_up_bound = np.array([[np.inf, np.inf, np.inf, np.inf,
                                        6.0,
                                        6.0, 4.0,
                                        5.0, 5.0, 5.0,
                                        0.3, 0.1, 0.4,
                                        0.1, 0.1, 0.5, 0.5]]).T
        self.actor.actor_wt[4:] += self.theta_low_bound[4:]
        self.actor.actor_wt = self.actor.actor_wt.clip(self.theta_low_bound, self.theta_up_bound)


    def state_to_feature(self, state):
        # e.g., s1^2+s2^2+s1s2+s1+s2+1
        SS = np.triu(np.outer(state, state))
        size = state.shape[0]
        phi_s = []
        for row in range(size):
            for col in range(row, size):
                phi_s.append(SS[row][col])
        phi_s = np.concatenate((phi_s, state, 1.0), axis=None)[:, None]
        return phi_s

    # return state value function V(s)
    def get_value(self, state):
        phi_S = self.state_to_feature(state)
        V = np.matmul(phi_S.T, self.critic_wt)[0, 0]
        return V

    # return act, info
    def get_action(self, state, act_wt=None, time=None, mode="train"):
        eps = self.eps if mode == "train" else 0.0
        act, info = self.actor.act_forward(state, act_wt=act_wt, time=time, mode=mode)  # act_wt = self.actor.actor_wt
        act += eps * (np.random.rand(self.action_dim))  # the added noise should de changed
        # act = act.clip(self.env.action_space.low, self.env.action_space.high)
        return act, info

    def train(self, replay_buffer: BasicBuffer):
        batch_size = min(self.batch_size, replay_buffer.size)
        train_it = min(self.iterations, int(3.0 * replay_buffer.size / batch_size))

        # Critic param update
        Av = np.zeros(shape=(self.critic_wt.shape[0], self.critic_wt.shape[0]))
        bv = np.zeros(shape=(self.critic_wt.shape[0], 1))
        for _ in tqdm_context(range(train_it), desc="Training Iterations"):
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(batch_size)

            for j, s in enumerate(states):
                S = self.state_to_feature(s)
                temp = (
                        S
                        - (1 - dones[j])
                        * self.gamma
                        * self.state_to_feature(next_states[j])
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
            states, actions, rewards, next_states, dones, infos = replay_buffer.sample(batch_size)

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

                try:
                    jacob_pi = self.actor.dPidP(
                        s, self.actor.actor_wt, info
                    )  # jacob:[n_a, n_sfeature]
                    psi = np.matmul(
                        jacob_pi.T, (actions[j] - pi_act)[:, None]
                    )  # psi: [n_sfeature, 1]

                    Aq += np.matmul(psi, psi.T)
                    bq += psi * (
                            rewards[j]
                            + (1 - dones[j]) * self.gamma * self.get_value(next_states[j])
                            - self.get_value(s)
                    )
                    G += np.matmul(jacob_pi.T, jacob_pi)
                except:
                    print(s)
                    print(actions[j])
                    print(next_states[j])
                    print(self.actor.actor_wt)
                    print(info)

                    soln = info["soln"]
                    x = soln["x"].full()
                    lam_g = soln["lam_g"].full()
                    z = np.concatenate((x, lam_g), axis=0)

                    self.actor.p_val[: self.obs_dim, 0] = s
                    self.actor.p_val[self.obs_dim:self.obs_dim + self.actor.theta_dim, :] = self.actor.actor_wt

                    # recall the time when calculate info
                    time = info["time"]
                    self.actor.p_val[self.obs_dim + self.actor.theta_dim:self.obs_dim + self.actor.theta_dim + self.actor.UNC_dim, :] = \
                        np.reshape(self.actor.env.uncertainty[:, time:time + self.actor.N], (-1, 1), order='F')
                    self.actor.p_val[self.obs_dim + self.actor.theta_dim + self.actor.UNC_dim:, :] = \
                        np.reshape(self.actor.env.price[:, time:time + self.actor.N], (-1, 1), order='F')

                    print(np.linalg.det(self.actor.drz(z, self.actor.p_val).full()))
                    # ping()


        if np.linalg.det(Aq) != 0.0:
            # update self.adv_wt
            print(np.linalg.det(Aq))
            self.adv_wt = np.linalg.solve(Aq, bq)

            # Policy param update
            if self.constrained_updates:
                self.actor.actor_wt = self.actor.param_update(
                    self.actor_lr / self.batch_size,
                    np.matmul(G, self.adv_wt),
                    self.actor.actor_wt,
                )
            else:
                self.actor.actor_wt -= (self.actor_lr / (train_it*batch_size)) * np.matmul(
                    G, self.adv_wt
                )
                self.actor.actor_wt = self.actor.actor_wt.clip(self.theta_low_bound, self.theta_up_bound)
            print("Params updated")
            print(self.actor.actor_wt.T)
        else:
            print("Rank deficient Aq")
            # print(Aq)
            print(self.actor.actor_wt.T)

        # print(self.critic_wt.T)
        # print(self.adv_wt.T)

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
        self.price_buy = csd.MX.sym('price_buy', 1)
        self.price_sell = csd.MX.sym('price_sell', 1)
        self.price = csd.vertcat(self.price_buy, self.price_sell)
        self.PRICE = csd.MX.sym("PRICE", self.price.size()[0], self.N)
        self.PRICE_dim = self.price.size()[0] * self.N
        # theta
        # dimension = 19
        self.theta_model = csd.MX.sym("theta_model", 4)
        self.theta_in = csd.MX.sym("theta_in", 1)
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

        # [Initial state=8, theta params=19, uncertainty=3*N, price=2*N]
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
        J = 0.0
        W = np.ones((1, self.sigma_dim))*1.0e2
        g = []  # Equality constraints
        hx = []  # Box constraints on states
        hu = []  # Box constraints on inputs
        hsg = []  # Box constraints on sigma

        # input inequalities
        # 0 < p_hp + theta_hp < 3 + sigma_hp
        hu.append(0.0 - self.U[4, 0])
        hu.append((self.U[4, 0] + self.theta_hp) - (3.0 + self.Sigma[3, 0]))
        # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
        hu.append((0.2 - self.Sigma[4, 0]) - (self.U[5, 0] + self.theta_xv))
        hu.append((self.U[5, 0] + self.theta_xv) - (0.8 + self.Sigma[4, 0]))
        # 0 < p_ch + theta_ch < 1 + sigma_ch
        hu.append(0.0 - self.U[0, 0])
        hu.append((self.U[0, 0] + self.theta_ch_dis[0]) - (1.0 + self.Sigma[6, 0]))
        # 0 < p_dis + theta_dis < 1 + sigma_dis
        hu.append(0.0 - self.U[1, 0])
        hu.append((self.U[1, 0] + self.theta_ch_dis[1]) - (1.0 + self.Sigma[7, 0]))
        # 0 < p_buy + theta_buy < 5 + sigma_buy
        hu.append(0.0 - self.U[2, 0])
        hu.append((self.U[2, 0] + self.theta_buy_sell[0]) - (4.0 + self.Sigma[8, 0]))
        # 0 < p_sell + theta_sell < 5 + sigma_sell
        hu.append(0.0 - self.U[3, 0])
        hu.append((self.U[3, 0] + self.theta_buy_sell[1]) - (4.0 + self.Sigma[9, 0]))

        # sys equalities: power balance
        g.append((self.UNC[1, 0] + self.U[4, 0] + self.U[0, 0] + self.U[3, 0]) -
                 (self.U[1, 0] + self.U[2, 0] + (self.beta * self.A_pv * self.UNC[0, 0])))
        # g.append((self.U[0, 0] * self.U[1, 0]))

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
            # g.append((self.U[0, i+1]*self.U[1, i+1]))

            # sys inequalities
            # 20 + sigma_t_1,2,3 < t_1,2,3 + theta_t_1,2,3 < 60 + sigma_t_1,2,3
            hx.append((20.0 - self.Sigma[0, i]) - (self.X[4, i] + self.theta_t[0]))
            hx.append((self.X[4, i] + self.theta_t[0]) - (60.0 + self.Sigma[0, i]))
            hx.append((20.0 - self.Sigma[1, i]) - (self.X[5, i] + self.theta_t[1]))
            hx.append((self.X[5, i] + self.theta_t[1]) - (60.0 + self.Sigma[1, i]))
            hx.append((20.0 - self.Sigma[2, i]) - (self.X[6, i] + self.theta_t[2]))
            hx.append((self.X[6, i] + self.theta_t[2]) - (60.0 + self.Sigma[2, i]))
            # 1 + sigma_e < e + theta_e< 4 + sigma_e
            hx.append((1.0 - self.Sigma[5, i]) - (self.X[7, i] + self.theta_e))
            hx.append((self.X[7, i] + self.theta_e) - (4.0 + self.Sigma[5, i]))

            # input inequalities
            # 0 < p_hp + theta_hp < 3 + sigma_hp
            hu.append(0.0 - self.U[4, i + 1])
            hu.append((self.U[4, i + 1] + self.theta_hp) - (3.0 + self.Sigma[3, i + 1]))
            # 0.2 + sigma_xv < p_xv + theta_xv < 0.8 + sigma_xv
            hu.append((0.2 - self.Sigma[4, i + 1]) - (self.U[5, i + 1] + self.theta_xv))
            hu.append((self.U[5, i + 1] + self.theta_xv) - (0.8 + self.Sigma[4, i + 1]))
            # 0 < p_ch + theta_ch < 1 + sigma_ch
            hu.append(0.0 - self.U[0, i + 1])
            hu.append((self.U[0, i + 1] + self.theta_ch_dis[0]) - (1.0 + self.Sigma[6, i + 1]))
            # 0 < p_dis + theta_dis < 1 + sigma_dis
            hu.append(0.0 - self.U[1, i + 1])
            hu.append((self.U[1, i + 1] + self.theta_ch_dis[1]) - (1.0 + self.Sigma[7, i + 1]))
            # 0 < p_buy + theta_buy < 5 + sigma_buy
            hu.append(0.0 - self.U[2, i + 1])
            hu.append((self.U[2, i + 1] + self.theta_buy_sell[0]) - (5.0 + self.Sigma[8, i + 1]))
            # 0 < p_sell + theta_sell < 5 + sigma_sell
            hu.append(0.0 - self.U[3, i + 1])
            hu.append((self.U[3, i + 1] + self.theta_buy_sell[1]) - (5.0 + self.Sigma[9, i + 1]))

            # slack inequalities
            for _ in range(self.sigma_dim):
                hsg.append(-self.Sigma[_, i])

        J += self.gamma ** (self.N - 1) * (
                self.terminal_cost(self.X[:, self.N - 1], self.theta) + W @ self.Sigma[:, self.N - 1])

        g.append(self.X[:, self.N - 1] - xn)
        hx.append((20.0 - self.Sigma[0, self.N - 1]) - (self.X[4, self.N - 1] + self.theta_t[0]))
        hx.append((self.X[4, self.N - 1] + self.theta_t[0]) - (60.0 + self.Sigma[0, self.N - 1]))
        hx.append((20.0 - self.Sigma[1, self.N - 1]) - (self.X[5, self.N - 1] + self.theta_t[1]))
        hx.append((self.X[5, self.N - 1] + self.theta_t[1]) - (60.0 + self.Sigma[1, self.N - 1]))
        hx.append((20.0 - self.Sigma[2, self.N - 1]) - (self.X[6, self.N - 1] + self.theta_t[2]))
        hx.append((self.X[6, self.N - 1] + self.theta_t[2]) - (60.0 + self.Sigma[2, self.N - 1]))
        hx.append((1.0 - self.Sigma[5, self.N - 1]) - (self.X[7, self.N - 1] + self.theta_e))
        hx.append((self.X[7, self.N - 1] + self.theta_e) - (4.0 + self.Sigma[5, self.N - 1]))
        for _ in range(self.sigma_dim):
            hsg.append(-self.Sigma[_, self.N - 1])


        # Constraints
        G = csd.vertcat(*g)
        Hu = csd.vertcat(*hu)
        Hx = csd.vertcat(*hx, *hsg)
        G_vcsd = csd.vertcat(*g, *hu, *hx, *hsg)

        lbg = [0] * G.shape[0] + [-math.inf] * Hu.shape[0] + [-math.inf] * Hx.shape[0]
        ubg = [0] * G.shape[0] + [0] * Hu.shape[0] + [0] * Hx.shape[0]
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
            "g": G_vcsd,
        }
        self.vsolver = csd.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)
        # self.dPi, self.dLagV = self.build_sensitivity(J, G, Hu, Hx)
        self.dR_sensfunc, self.dPi, self.drz = self.build_sensitivity(J, G, csd.vcat([Hu, Hx]))

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
        dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim+self.theta_dim]
        dPi = csd.Function("dPi", [z, self.P], [dzdP[:self.action_dim, :]])

        return dR_sensfunc, dPi, drz

    def stage_cost_fn(self):
        l_spo = self.price_buy * self.u[2] - self.price_sell * self.u[3]
        l_tem = self.theta_in[0] * (self.x[1] - 23) * (self.x[1] - 27)
        stage_cost = l_tem + l_spo
        stage_cost_fn = csd.Function("stage_cost_fn", [self.x, self.u, self.theta, self.price], [stage_cost])
        return stage_cost_fn

    def terminal_cost_fn(self):
        terminal_cost = self.theta_en[0] ** 2 * (self.x[7] - self.theta_en[1]) ** 2
        terminal_cost_fn = csd.Function("stage_cost_fn", [self.x, self.theta], [terminal_cost])
        return terminal_cost_fn


class Custom_MPCActor(Custom_QP_formulation):
    def __init__(self, env, mpc_horizon, cost_params, gamma=1.0, debug=False):
        upper_tri = cost_params["upper_tri"] if "upper_tri" in cost_params else False
        super().__init__(env, mpc_horizon, gamma, cost_params["cost_defn"], upper_tri)
        self.debug = debug
        self.p_val = np.zeros((self.p_dim, 1))

        self.actor_wt = np.array(cost_params["cost_wt"]) if "cost_wt" in cost_params \
            else 0.01 * np.random.rand(self.theta_dim, 1)

        # Test run
        _ = self.act_forward(self.env.reset())
        self.X0 = None
        self.soln = None
        self.info =None

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

        # [dRdz, dRdP] = self.dR_sensfunc(z, self.p_val)
        # dzdP = (-np.linalg.solve(dRdz, dRdP[:, self.obs_dim:self.obs_dim+self.theta_dim]))
        # dzdP = -csd.inv(dRdz) @ dRdP[:, self.obs_dim:self.obs_dim + self.theta_dim]
        # dpi = dzdP[:self.action_dim, :]

        dpi = self.dPi(z, self.p_val).full()
        return dpi

    def param_update(self, lr, dJ, act_wt):
        print("Not implemented: constrained param update")
        return act_wt
