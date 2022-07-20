import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from helpers import tqdm_context
from Agents.abstract_agent import TrainableController


def exp_init():
    cmd_line = sys.argv
    with open(cmd_line[1]) as f:
        params = json.load(f)
        print(params)

    # Results directory
    results_dir = os.path.join("./Results", params["env"], params["results_dir"])
    if params["save_data"]:
        print(results_dir, end="\n\n")
        os.makedirs(results_dir, exist_ok=True)
        with open(results_dir + "/setting.json", "w") as fp:
            json.dump(params, fp, indent=4)

    # Seeding
    seed = params["seed"]
    return params, results_dir, seed


def iter_init(it, data_dict):
    print(f"Iteration: {it}")
    data_dict.update({it: {}})


def train_agent(agent, replay_buffer, info_dict):
    if isinstance(agent, TrainableController):
        add_info = agent.train(replay_buffer)
        info_dict.update(add_info)


def process_iter_returns(
    it,
    results_dir,
    data_dict,
    replay_buffer,
    train_returns,
    eval_returns,
    t_returns,
    e_returns,
    save_data=True,
):
    # Iteration results
    train_returns.append(mean(t_returns))
    print(f"Training return: {mean(t_returns)}")

    mean_e = mean(t_returns)
    if e_returns:
        mean_e = mean(e_returns)
    eval_returns.append(mean_e)
    print(f"Evaluation return: {mean_e}", end="\n\n")

    # Logging results
    data_dict.update({"train_returns": train_returns})
    data_dict.update({"eval_returns": eval_returns})
    data_dict[it].update(
        {
            "eval_return": mean_e,
            "e_returns": e_returns,
            "train_return": mean(t_returns),
            "t_retruns": t_returns,
        }
    )
    if save_data:
        with open(results_dir + "/data_dict.npy", "wb") as fp:
            np.save(fp, data_dict)
        with open(results_dir + "/replay_buffer.pkl", "wb") as fp:
            # np.save(fp, replay_buffer.buffer)
            pickle.dump(replay_buffer.buffer, fp)


def plot_eval_perf(results_dir, params, eval_returns, save_data=True):
    print(params)
    print(eval_returns)

    # plotting scripts
    perf = plt.figure("Evaluation Performance")
    plt.plot(eval_returns)
    plt.xlabel("iterations")
    plt.ylabel("J")
    if save_data:
        plt.savefig(results_dir + "/" + "eval.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close(perf)


def rollout_sample(env, agent, replay_buffer, n_step, mode="train"):
    state, obs = env.reset()
    rollout_return = 0

    for _ in tqdm_context(range(n_step), desc="Episode", pos=1):
        # print(f' evaluation step {step}-------------')
        action, add_info = agent.get_action(obs, time=env.t, mode=mode)
        next_state, next_obs, reward, done = env.step(action)
        if mode == "train" or mode == "mpc":
            replay_buffer.push(
                state, obs, action, reward, next_state, next_obs, done, add_info
            )
            # print([state, action])
            # print(reward)

        if mode == "final":
            print([state, action])
            print(reward)

        rollout_return += reward
        state = next_state.copy()
        obs = next_obs.copy()
    return rollout_return
