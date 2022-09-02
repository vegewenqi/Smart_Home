import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plotter:
    def __init__(self, open_file_name, save_fig_name):
        self.open_file_name = open_file_name
        self.save_fig_name = save_fig_name
        self.num_theta = 15
        self.num_step = 1010

        self.all_step = []
        self.all_theta = [[] for i in range(self.num_theta)]

        self.legends = [r"$\theta_{\rm{m1}}$", r"$\theta_{\rm{m2}}$", r"$\theta_{\rm{m3}}$", r"$\theta_{\rm{m4}}$",
                        r"$\theta_{\rm{m5}}$", r"$\theta_{\rm{m6}}$", r"$\theta_{\epsilon2}$",
                        r"$\theta_{\epsilon3}$", r"$\theta_{\rm{in1}}$", r"$\theta_{\rm{in2}}$",
                        r"$\theta_{\rm{in3}}$", r"$\theta_{\rm{en1}}$", r"$\theta_{\rm{en2}}$",
                        r"$\theta_{\rm{t2}}$", r"$\theta_{\rm{e}}$"]

        for file_name in self.open_file_name:
            f = open(file_name, "rb")
            results = pickle.load(f)
            f.close()

            # self.all_step.extend(list(range(len(results['theta_log']))))
            self.all_step.extend(list(range(self.num_step)))

            theta = np.zeros((self.num_theta, self.num_step))
            for ite in range(self.num_step):
                for index_theta in range(self.num_theta):
                    theta[index_theta, ite] = results['theta_log'][ite][index_theta]

            for i in range(self.num_theta):
                self.all_theta[i].extend(theta[i])

        data_dict = {'Steps': self.all_step}
        for i in range(self.num_theta):
            data_dict[self.legends[i]] = self.all_theta[i]
        self.df = pd.DataFrame(data_dict)

    def plot(self):
        fig, ax = plt.subplots(3, 5)
        for i in range(15):
            # ax1[i // 5, i % 5].plot(self.theta_step, self.theta[i, :1010])
            g = sns.lineplot(x="Steps", y=self.legends[i], data=self.df, ax=ax[i // 5, i % 5])
            ax[i // 5, i % 5].set_xlim((0, self.num_step))
            ax[i // 5, i % 5].ticklabel_format(useOffset=False, axis='y')
            ax[i // 5, i % 5].legend(labels=[self.legends[i]])
            g.set(ylabel=None)
        ax[2, 3].set_ylim((-0.001, 0.001))

        ### one legend for all sugplots
        plt.tight_layout()
        plt.show()
        # plt.savefig(self.save_fig_name)


if __name__ == '__main__':
    Plotter = Plotter(['results_rl_try_0.pkl',
                       'results_rl_try_1.pkl',
                       'results_rl_try_2.pkl',
                       'results_rl_try_3.pkl',
                       'results_rl_try_4.pkl'],
                      'results_rl_fig_try_0.png')
    Plotter.plot()
