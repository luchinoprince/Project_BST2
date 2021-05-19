import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import random
from collections import Counter
from operator import itemgetter

"""
    DPModel: Generate exchangeable sequence from Dirichlet Process and produce a graph with trajectories, as well as credible intervals. This model is flexible to test the impact of some arguments to the result, e.g. sample size, alpha value, and easy to be extended to support other update, like base distribution.

    Arguments:
        n_data: number of data to be generated for each trajectory.
        n_traj: number of trajectories.
        alpha: dirichlet process concentration parameter.
        F_base: dirichlet process base distribution. It's required to have .cdf() and .rvs() function, it's better to come from scipy.stats package.
        fig_name: output figure file name

    run_simulation(): takes the default parameters, output a graph with n_traj number of trajectories, with 95% credible intervals bands

    run_simulation_with_true_data():
        Takes two additional arguments:
            F_true: true distribution that we generate true data from
            data_size: number of true data we generate to run the model
        Output a graph with trajectories and credible interval bands, as well as cdf of true distribution and Dirichelt process base distribution.

    How to run:
        1. Initiate Dirichlet Process simulator with basic settings:
          dp_model = DPModel(n_data = 500, n_traj = 50, alpha = 80, F_base = st.uniform, fig_name = "DP80_1.png")
        2. Run simulation with default settings:
          dp_model.run_simulation()
        3. Generate data from some true distribution and run simulation: 
          dp_model.run_simulation_with_true_data(F_true = st.cauchy, data_size = 50)
        4. Update alpha to run another simulation:
          dp_model.update_alpha(alpha = 50)
          dp_model.update_fig_name(figname = "MyNewFigure.png")
          dp_model.run_simulation() 
"""

class DPModel:
    def __init__(self, n_data, n_traj, alpha, F_base, fig_name):
        self.n_data = n_data
        self.n_traj = n_traj
        self.alpha = alpha
        self.F_base = F_base
        self.true_data = []
        self.F_true = None
        self.fig_name = fig_name

    def update_alpha(self, alpha):
        self.alpha = alpha

    def update_fig_name(self, fig_name):
        self.fig_name = fig_name
    
    def update_num_of_data(self, n_data):
        self.n_data = n_data

    def __generate_true_data(self, F_true, data_size):
        self.true_data = list(F_true.rvs(size = data_size))
        self.F_true = F_true

    def __plot_cdf(self, traj_list):
        fig, ax = plt.subplots(figsize = (15,10))

        plt.title("Realization of DPs, with alpha: " + str(self.alpha))

        # draw trajectories
        for p in traj_list:
            ax.step([i for i,j in p], np.cumsum([j for i,j in p]) / (self.n_data + len(self.true_data)), alpha = 0.6)

        # draw confidence marginal interval 
        T = np.linspace((min(min(traj_list, key = lambda x: x[0])))[0], (max(max(traj_list, key = lambda x: x[0])))[0], 1000)
        num_true_data = len(self.true_data)    

        conf_bands = []

        for t in T:
            if num_true_data == 0:
                base = self.F_base.cdf(t)
            else:
                base = self.alpha/(self.alpha + num_true_data) * self.F_base.cdf(t) + np.sum([ t >= a for a in self.true_data])/(self.alpha + num_true_data)

            conf_bands.append((st.beta.ppf(0.025, (self.alpha + num_true_data) * base, (1 - base) * (self.alpha + num_true_data)), st.beta.ppf(0.975, (self.alpha + num_true_data) * base, (1 - base) * (self.alpha + num_true_data))))

        plt.fill_between(T, [a[0] for a in conf_bands], [a[1] for a in conf_bands], color='r', alpha=.1, label="95% marginal confidence region") 
        plt.legend()    

        # draw true data generation distribution and base distribution
        if num_true_data != 0:
            plt.plot(T, self.F_true.cdf(T),'-', label = "True data generating distribution")
            plt.plot(T, self.F_base.cdf(T),'-', label = "F_Base")

        plt.savefig(self.fig_name)
        
    def run_simulation(self):
        traj_list = []

        # generate exchangeable sequence using Blackwell-MacQuenn Polya Urn scheme
        for t in range(self.n_traj):
            x = [self.F_base.rvs()]
            x += self.true_data

            for n in range(1, self.n_data):
                u = np.random.rand()
                if u < self.alpha / (self.alpha + n):
                    x.append(self.F_base.rvs())
                else:
                    x.append(x[st.randint.rvs(0,n + len(self.true_data))])
            traj_list.append(sorted(list(Counter(x).items()), key = itemgetter(0)))

        self.__plot_cdf(traj_list)

    def run_simulation_with_true_data(self, F_true, data_size):
        self.__generate_true_data(F_true, data_size)
        self.run_simulation()

# Exercise 1:
ex_1 = DPModel(n_data = 500, n_traj = 50, alpha = 80, F_base = st.uniform, fig_name = "DP80_1.png")
alpha_list = [1, 10, 70, 80]
for alpha in alpha_list:
    fig_name = "DP" + str(alpha) + "_1.png"
    ex_1.update_alpha(alpha)
    ex_1.update_fig_name(fig_name)
    ex_1.run_simulation()

# Exercise 2 with base_distribution N(0, 3), and true data generated from Cauchy distribution
ex_2 = DPModel(n_data = 500, n_traj = 50, alpha = 80, F_base = st.norm(0, 3), fig_name = "DP80_2.png")
alpha_datasize_list = [(10, 20), (10, 100), (100, 100), (500, 100)]
for pair in alpha_datasize_list:
    fig_name = "DP" + str(pair[0]) + "_" + str(pair[1]) + "_2.png"
    ex_2.update_fig_name(fig_name)
    ex_2.update_alpha(pair[0])
    ex_2.run_simulation_with_true_data(F_true = st.cauchy, data_size = pair[1])

