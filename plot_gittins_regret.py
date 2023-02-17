import numpy as np
from gittins import *
from gdmax import gittins_GD_max

""" Plotting Bayes-optimal utilities and/or regret of the Bayes-optimal policy when one arm in a 2-armed Bernoulli bandit is fixed """

# compute (offline) Gittins indices for belief states (up to num_actions)
gittins_table = bmab_gi_multiple_ab(alpha_start=1, beta_start=1, gamma=0.9, N=100, num_actions=70, tol=5e-9)

# print("Beta(1,1), Beta(1,1)", get_Bayes_exp_regret_of_Bayes_opt_pol(gittins_table, prior_alphas=[1, 1], prior_betas=[1, 1]))
# print("Beta(1,1), Beta(1,1)", get_Bayes_exp_regret_of_Bayes_opt_pol(gittins_table, prior_alphas=[1, 1], prior_betas=[1, 1]))
# print("Beta(3, 3), Beta(3, 3)", get_Bayes_exp_regret_of_Bayes_opt_pol(gittins_table, prior_alphas=[3, 3], prior_betas=[3, 3]))
# print("Beta(2, 2), Beta(1, 1)", get_Bayes_exp_regret_of_Bayes_opt_pol(gittins_table, prior_alphas=[2, 2], prior_betas=[1, 1]))

# plot Bayes-optimal utility with Binomial parameterisation and fixed p
# plot_bayes_opt_ut_binom_param_fixed_p(gittins_table, p_nom=2, p_denom=3)

# plot Bayes-optimal utility with Binomial parameterisation and fixed n
# plot_bayes_opt_ut_binom_param_fixed_n(gittins_table, fixed_alpha=1, fixed_beta=4, n=10)

# plot Bayes-optimal utility/regret landscape
plot_bayes_opt_ut_2_armed_bandit(gittins_table, fixed_alpha=2, fixed_beta=4, alpha_range=10, beta_range=10)

""" Computing approximately maximin prior using GDMax algorithm """

# maximin_alphas, maximin_betas = gittins_GD_max()
