from importlib import import_module


def agent_init(env, agent_str, agent_params):
    agent_class = AgentFactory(agent_str=agent_str)
    agent = agent_class(env, agent_params)
    return agent


class AgentFactory:
    valid_agents = {
        "TD3": [".td3", "TD3Agent"],
        "SAC": [".sac", "SACAgent"],
        "RLMPC_LSTDQ": [".rl_mpc_lstdq", "RLMPC_LSTDQ_Agent"],
        "Qlearning_MPC": [".qlearning_mpc", "Qlearning_MPC_Agent"],
        "QPlearning_MPC": [".qplearning_mpc", "QPlearning_MPC_Agent"],
        "MPC_AC": [".mpc_ac", "MPC_AC_Agent_exp"],
        "SmartHome_RLMPC_LSTD": [".smart_home_rl_mpc_lstd", "Smart_Home_MPCAgent"],
        "House4Pumps_RLMPC_LSTD": [".house_4pumps_rl_mpc_lstd", "House_4Pumps_MPCAgent"],
        "SmartHome_RLMPC_CDDAC": [".smart_home_rl_mpc_cddac", "Smart_Home_MPCAgent"],
        "SmartHome_RLMPC_CDDAC_FAST": [".smart_home_rl_mpc_cddac_fast", "Smart_Home_MPCAgent"],
        "SmartHome_RLMPC_CDDAC_FAST_TRY": [".smart_home_rl_mpc_cddac_fast_try", "Smart_Home_MPCAgent"],
    }
    agent = None

    def __new__(cls, *, agent_str):
        if agent_str in cls.valid_agents:
            agent_package, agent_class = cls.valid_agents[agent_str]
            module = import_module(agent_package, "Agents")
            cls.agent = getattr(module, agent_class)
        else:
            raise ImportError(f"{agent_str} not implemented/known to agent factory")
        return cls.agent
