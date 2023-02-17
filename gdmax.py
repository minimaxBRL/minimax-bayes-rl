import numpy as np
from gittins import *
from scipy.special import psi


# GD-Max with policy oracle using the Gittins index
# Returns an approximately maximin product Beta prior
def gittins_GD_max(learning_rate=0.8, n_iterations=250, n_arms=2):
    # initial prior
    alphas = np.ones(n_arms)
    betas = np.ones(n_arms)

    for i in range(n_iterations):
        step_alphas, step_betas, regret = gradient_step(alphas, betas, n_arms)
        print("Iteration", i, "Belief", alphas, betas, "Regret", regret)
        alphas += learning_rate * step_alphas
        betas += learning_rate * step_betas
        alphas[alphas <= 0] = 0.01
        betas[betas <= 0] = 0.01

    return alphas, betas


def gradient_step(alphas, betas, n_arms, n_samples=300):
    step_alphas = np.zeros(n_arms)
    step_betas = np.zeros(n_arms)

    gittins_indices = []

    avg_regret = 0

    # for every arm, we need to compute the Gittins index (starting form that arm's belief state)
    for i in range(n_arms):
        gittins_depth = 40
        gittins_indices.append(bmab_gi_multiple_ab(alpha_start=alphas[i], beta_start=betas[i], gamma=0.9, N=100, num_actions=gittins_depth, tol=5e-5))

    for _ in range(n_samples):
        # sample Bernoulli bandit from prior
        bandit = np.random.beta(alphas, betas, n_arms)

        # compute the Bayes-expected regret for this bandit
        regret = bayes_exp_regret_of_sampled_bandit(bandit, gittins_indices, gittins_depth, alphas, betas, n_arms)

        step_alphas += regret * (psi(np.array(alphas) + np.array(betas)) - psi(alphas) + np.log(bandit))
        step_betas += regret * (psi(np.array(alphas) + np.array(betas)) - psi(betas) + np.log(1 - np.array(bandit)))

        avg_regret += regret

    return step_alphas / n_samples, step_betas / n_samples, avg_regret / n_samples


# Compute regret of the Bayes-optimal policy for a sampled bandit
def bayes_exp_regret_of_sampled_bandit(bandit, gittins_indices, gittins_depth, alphas, betas, n_arms, n_rollouts=300, gamma=0.9):
    avg_bayes_opt_ut = 0

    for _ in range(n_rollouts):

        belief_state = [[0, 0] for arm in range(n_arms)]
        ut = 0
        for n in range(gittins_depth - 2):

            # lookup arm indices (axis 0 of gittins corresponds to betas, axis 1 to alphas)
            arm_indices = []

            for arm in range(n_arms):
                arm_indices.append(gittins_indices[arm][int(belief_state[arm][1])][int(belief_state[arm][0])])
            action = np.argmax(arm_indices)
            obs = np.random.binomial(1, bandit[action])
            ut += obs * (gamma ** n)

            # update belief state
            belief_state[action][0] += obs
            belief_state[action][1] += 1 - obs

        avg_bayes_opt_ut += ut

    avg_bayes_opt_ut /= n_rollouts

    return max(bandit) * (1 / (1 - gamma)) - avg_bayes_opt_ut
