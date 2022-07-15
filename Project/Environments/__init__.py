from importlib import import_module


def env_init(env_str, env_params, seed):
    return EnvFactory(env_str, env_params, seed)


class EnvFactory:
    valid_envs = {
        "pendulum": ["gym", "Pendulum-v0"],
        "point_mass": [".point_mass", "PointMass"],
        "motor_model": [".motor_model", "MotorContinuousEnv"],
        "smart_home": [".smart_home", "SmartHome"],
        "smart_home_simplified": [".smart_home_simplified", "SmartHome_simplified"],
        "house_4pumps": [".house_4pumps", "House4Pumps"],
        "smart_home_plot": [".smart_home_plot", "SmartHome"],
    }
    modified_envs = ["point_mass", "motor_model", "smart_home", "smart_home_simplified",
                     "house_4pumps", "smart_home_plot"]
    env = None

    def __new__(cls, env_str, env_params, seed):
        if env_str in cls.valid_envs:
            env_package, env_class = cls.valid_envs[env_str]
            if env_str in cls.modified_envs:
                module = import_module(env_package, "Environments")
                cls.env = getattr(module, env_class)(env_str, env_params, seed)
            elif env_package == "gym":
                import gym

                cls.env = gym.make(env_class)
            else:
                raise ImportError(
                    f"{env_str} is present in env factory but not modified"
                )
        else:
            raise ImportError(f"{env_str} not implemented/known to env factory")
        return cls.env
