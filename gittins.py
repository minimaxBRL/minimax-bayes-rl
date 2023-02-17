import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import numpy as np
import copy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

rpy2.robjects.numpy2ri.activate()

gittins = importr('gittins')
gittins = importr('gittins')


def bmab_gi_multiple_ab(alpha_start, beta_start, gamma, N, num_actions, tol=5e-5):
    return gittins.bmab_gi_multiple_ab(alpha_start, beta_start, gamma, N, num_actions, tol)


def nmab_gi_multiple(n_range, gamma, tau, tol, N, xi, delta):
    raise NotImplementedError  # check n_range arg
    return gittins.nmab_gi_multiple(n_range, gamma, tau, tol, N, xi, delta)


# Compute (approximately) Bayes-optimal utility given a product Beta prior (over all arms); discounted infinite horizon
# param list prod_beta_prior: product Beta prior as list of [alpha_i, beta_i] parameter pairs, length of list = no. of arms
# param list gittins_idx: precomputed Gittins indices
# param int n_rollouts: number of rollouts to approximate utility
# param float gamma: discount factor
def bmab_bayes_opt_ut_via_rollouts(gittins_idx, prior_alphas=None, prior_betas=None, n_rollouts=10000, gamma=0.9):
    assert len(prior_alphas) == len(prior_betas)
    # rollout horizon, ensure that belief state does not exceed Gittins table
    h_rollouts = len(gittins_idx) - 15

    n_arms = len(prior_alphas)
    avg_bayes_opt_ut = 0

    for _ in range(n_rollouts):
        # sample bandit from prior as list of Bernoulli parameters
        bandit = np.random.beta(prior_alphas, prior_betas, n_arms)
        belief_state = [[prior_alphas[arm] - 1, prior_betas[arm] - 1] for arm in range(n_arms)]

        ut = 0
        for n in range(h_rollouts):
            # make sure we have the Gittins index for the current belief state
            assert np.max(belief_state) < len(gittins_idx)
            # lookup arm indices (axis 0 of gittins_idx corresponds to betas, axis 1 to alphas)
            arm_indices = [gittins_idx[bel_arm[1]][bel_arm[0]] for bel_arm in belief_state]
            action = np.argmax(arm_indices)
            obs = np.random.binomial(1, bandit[action])
            ut += obs * (gamma ** n)
            # update belief state
            belief_state[action][0] += obs
            belief_state[action][1] += 1 - obs

        avg_bayes_opt_ut += ut
    return avg_bayes_opt_ut / n_rollouts


# Having a look at the Bayes-optimal utility landscape for 2-armed Bernoulli bandits; we fix the prior Beta(a,b) for one arm
# x_axis: alphas, y_axis: betas, z_axis: utility
def plot_bayes_opt_ut_2_armed_bandit(gittins_idx, fixed_alpha=1, fixed_beta=1, alpha_range=10, beta_range=10):
    ut = np.zeros([alpha_range, beta_range])
    ut_oracle = np.zeros([alpha_range, beta_range])
    for alpha in range(0, alpha_range):
        print("Alpha", alpha)
        for beta in range(0, beta_range):
            alphas = [alpha + 1, fixed_alpha]
            betas = [beta + 1, fixed_beta]
            # Bayes-optimal utility
            ut[alpha, beta] = bmab_bayes_opt_ut_via_rollouts(gittins_idx, prior_alphas=alphas, prior_betas=betas)
            # Utility of an oracle
            ut_oracle[alpha, beta] = get_utility_of_oracle(alphas, betas)
    fig = plt.figure(dpi=300)
    # ax = plt.axes(projection='3d')
    x = np.linspace(1, alpha_range, alpha_range)
    y = np.linspace(1, beta_range, beta_range)
    x, y = np.meshgrid(x, y)

    # plot Bayes-optimal utility
    # ax.plot_surface(x, y, ut, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # plot Bayes-expected regret
    # ax.plot_surface(x, y, ut_oracle - ut, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.imshow(ut_oracle - ut)
    plt.colorbar()

    # don't know why, but apparently x-axis is betas and y-axis alphas
    plt.ylabel('alphas')
    plt.xlabel('betas')
    plt.xticks(range(0, 10), range(1, 11), fontsize=12)
    plt.yticks(range(0, 10), range(1, 11), fontsize=12)

    # ax.set_zlabel('regret')
    ax = plt.gca()
    ax.invert_yaxis()

    # # get worst-case prior for the first arm
    # regret = ut_oracle - ut
    # worst_case_prior = np.unravel_index(regret.argmax(), regret.shape)
    # worst_case_regret = regret[worst_case_prior]
    # print("Regret of Beta(1,1)", regret[0, 0])
    # print("Regret of Matching Fixed Arm", regret[fixed_alpha - 1, fixed_beta - 1])
    # print("Maximum Regret", worst_case_regret)
    # print("Worst-Case Prior", [worst_case_prior[0] + 1, worst_case_prior[1] + 1])

    plt.savefig('regret_landscape_09_' + str(fixed_alpha) + '_' + str(fixed_beta) + '.png', dpi=600, bbox_inches='tight')
    plt.show()

    return ut


# Plot the Bayes-optimal utility for a 2-armed Bernoulli bandit with n and p (binomial) parameterisation.
# The parameterisation corresponds to n=alpha+beta and p=alpha/(alpha+beta)
# param int n: number of pulls; this is fixed and we vary the parameter p for this fixed n
def plot_bayes_opt_ut_binom_param_fixed_n(gittins_idx, fixed_alpha=1, fixed_beta=1, n=10):
    # the range of alpha and beta values is given by n-1
    alpha_beta_range = n - 1
    ut = np.zeros(alpha_beta_range)
    ut_oracle = np.zeros(alpha_beta_range)
    for alpha in range(alpha_beta_range):
        alphas = [alpha + 1, fixed_alpha]
        betas = [n - (alpha + 1), fixed_beta]
        ut[alpha] = bmab_bayes_opt_ut_via_rollouts(gittins_idx, prior_alphas=alphas, prior_betas=betas)
        ut_oracle[alpha] = get_utility_of_oracle(alphas, betas)
    p_values = [(alpha + 1) / n for alpha in range(alpha_beta_range)]
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(p_values, ut)
    ax1.set_xlabel('p')
    ax1.set_ylabel('Bayes-optimal Utility')
    ax2.plot(p_values, ut_oracle)
    ax2.set_xlabel('p')
    ax2.set_ylabel('Oracle Utility')
    ax3.plot(p_values, ut_oracle - ut)
    ax3.set_xlabel('p')
    ax3.set_ylabel('Bayes-expected Regret')
    plt.show()
    return ut


# Plot the Bayes-optimal utility for a 2-armed Bernoulli bandit with n and p (binomial) parameterisation.
# The parameterisation corresponds to n=alpha+beta and p=alpha/(alpha+beta)
# Parameters p_nom and p_denom result in a fixed p = p_nom/p_denom and we vary n (as multiples of p_denom)
# param int p_nom: the nominator of p
# param int p_denom: the denominator of p
def plot_bayes_opt_ut_binom_param_fixed_p(gittins_idx, fixed_alpha=1, fixed_beta=1, p_nom=2, p_denom=3):
    n_values = [p_denom * i for i in range(1, 7)]
    alpha_values = [p_nom * i for i in range(1, 7)]
    beta_values = [n_values[i] - alpha_values[i] for i in range(len(n_values))]
    ut = np.zeros(len(n_values))
    ut_oracle = np.zeros(len(n_values))
    for i in range(len(n_values)):
        alphas = [alpha_values[i], fixed_alpha]
        betas = [beta_values[i], fixed_beta]
        ut[i] = bmab_bayes_opt_ut_via_rollouts(gittins_idx, prior_alphas=alphas, prior_betas=betas)
        ut_oracle[i] = get_utility_of_oracle(alphas, betas)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(n_values, ut)
    ax1.set_xlabel('n')
    ax1.set_ylabel('Bayes-optimal Utility')
    ax2.plot(n_values, ut_oracle)
    ax2.set_xlabel('n')
    ax2.set_ylabel('Oracle Utility')
    ax3.plot(n_values, ut_oracle - ut)
    ax3.set_xlabel('n')
    ax3.set_ylabel('Bayes-expected Regret')
    plt.show()
    return ut


# To get the oracle's utility we need to compute E[max(X_1, ..., X_k)], where X_i is Beta(alpha_i, beta_i)-distributed
# I am not aware of a closed-form solution for this, so we approximate
def get_utility_of_oracle(prior_alphas, prior_betas):
    oracle = 0
    n = 10000
    for _ in range(n):
        bandit = np.random.beta(prior_alphas, prior_betas)
        oracle += max(bandit)
    return (oracle / n) * (1 / (1 - 0.9))


def get_Bayes_exp_regret_of_Bayes_opt_pol(gittins_idx, prior_alphas, prior_betas, n_rollouts=20000):
    n_arms = len(prior_alphas)
    avg_bayes_opt_ut = 0
    for _ in range(n_rollouts):
        # sample bandit from prior as list of Bernoulli parameters
        bandit = np.random.beta(prior_alphas, prior_betas, n_arms)
        belief_state = [[prior_alphas[arm] - 1, prior_betas[arm] - 1] for arm in range(n_arms)]
        ut = 0
        for n in range(len(gittins_idx) - 5):
            # make sure we have the Gittins index for the current belief state
            assert np.max(belief_state) < len(gittins_idx)
            # lookup arm indices (axis 0 of gittins_idx corresponds to betas, axis 1 to alphas)
            arm_indices = [gittins_idx[bel_arm[1]][bel_arm[0]] for bel_arm in belief_state]
            action = np.argmax(arm_indices)
            obs = np.random.binomial(1, bandit[action])
            ut += obs * (0.9 ** n)
            # update belief state
            belief_state[action][0] += obs
            belief_state[action][1] += 1 - obs
        avg_bayes_opt_ut += ut
    avg_bayes_opt_ut /= n_rollouts
    return get_utility_of_oracle(prior_alphas, prior_betas) - avg_bayes_opt_ut
