from Environments import env_init
from Agents import agent_init
from replay_buffer import BasicBuffer
from helpers import tqdm_context
from rollout_utils import exp_init, iter_init, rollout_sample, train_agent, \
    process_iter_returns, plot_eval_perf

# Experiment init
params, results_dir, seed = exp_init()

# Environment, agent and buffer initialization
env = env_init(params["env"], params["env_params"], seed)
agent = agent_init(env, params["agent"], params["agent_params"], seed)
replay_buffer = BasicBuffer(params["buffer_maxlen"], seed)

# Returns
train_returns = []
eval_returns = []
data_dict = {}

for it in tqdm_context(range(params["n_iterations"]), desc="Iterations", pos=3):
    iter_init(it, data_dict)

    # Sampling
    t_returns = []
    for train_runs in tqdm_context(range(params["n_trains"]), desc="Train Rollouts"):
        rollout_return = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="train")
        t_returns.append(rollout_return)

    # Replay + Learning
    train_agent(agent, replay_buffer, data_dict[it])

    # Evaluation
    e_returns = []
    if (it + 1) % 1 == 0:
        for eval_runs in tqdm_context(range(params["n_evals"]), desc="Evaluation Rollouts"):
            rollout_return = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="eval")
            e_returns.append(rollout_return)

    process_iter_returns(
        it,
        results_dir,
        data_dict,
        replay_buffer,
        train_returns,
        eval_returns,
        t_returns,
        e_returns,
        params["save_data"],
    )

# Final rollout for visualization
_ = rollout_sample(env, agent, replay_buffer, params["n_steps"], mode="final")

# Evaluation performance plot
plot_eval_perf(results_dir, params, eval_returns)

