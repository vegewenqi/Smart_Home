import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


class Plotter:
    def __init__(self, open_file_names, save_fig_name):
        self.open_file_names = open_file_names
        self.save_fig_name = save_fig_name
        self.eval_returns = []
        self.eval_step = []

        for file_name in self.open_file_names:
            f = open(file_name, "rb")
            results = pickle.load(f)
            f.close()
            self.eval_returns.extend(results['eval_returns'][:101])
            self.eval_step.extend(10 * np.arange(len(results['eval_returns'][:101])))

        self.df = pd.DataFrame(dict(Steps=self.eval_step,
                                    Performance=self.eval_returns))

    def plot(self):
        g = sns.relplot(x="Steps", y="Performance", kind="line", data=self.df)
        g.set(xlabel="Learning steps", ylabel="Performance")
        plt.xlim((0, 1000))
        # plt.title(self.save_fig_name)
        plt.savefig(self.save_fig_name)
        plt.legend(labels=[r"$J(\pi_{\theta})$", "95% Confidence interval"])
        plt.show(block=True)


if __name__ == '__main__':
    Plotter = Plotter(['results_rl_try_0.pkl', 'results_rl_try_1.pkl', 'results_rl_try_2.pkl',
                       'results_rl_try_3.pkl', 'results_rl_try_4.pkl'],
                      'Learning steps vs Performance.pdf')
    Plotter.plot()
