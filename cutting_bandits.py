from cuttingplane import cutting_plane, get_initial_constraints_beta_ab, plot_all
from gittins import bmab_gi_multiple_ab
import numpy as np
from scipy.special import digamma

dim = 2
epsilon = 0.0
delta = 0.001
gamma = 0.95
n_rollouts = 10000

constraints, b_ub = get_initial_constraints_beta_ab()


def bmab_bayes_opt_ut_via_rollouts(
    gittins_idx, prior_alphas=None, prior_betas=None, n_rollouts=5000, gamma=0.9, reparam=False
):
    assert prior_alphas.shape == prior_betas.shape
    # rollout horizon, ensure that belief state does not exceed Gittins table
    h_rollouts = len(gittins_idx)

    n_arms = 2
    avg_bayes_opt_ut = 0
    avg_regret = 0
    gradient = np.zeros(2)
    for _ in range(n_rollouts):
        # sample bandit from prior as list of Bernoulli parameters
        bandit = np.random.beta(prior_alphas, prior_betas, n_arms)
        belief_state = [[0, 0] for arm in range(n_arms)]

        ut = 0
        for n in range(h_rollouts):
            # make sure we have the Gittins index for the current belief state
            # lookup arm indices (axis 0 of gittins_idx corresponds to betas, axis 1 to alphas)
            arm_indices = [
                gittins_idx[bel_arm[1]][bel_arm[0]] for bel_arm in belief_state
            ]
            action = np.argmax(arm_indices)
            obs = np.random.binomial(1, bandit[action])
            ut += obs * (gamma ** n)
            # update belief state
            belief_state[action][0] += obs
            belief_state[action][1] += 1 - obs

        avg_bayes_opt_ut += ut
        regret = (1 - gamma ** h_rollouts) / (1 - gamma) * np.max(bandit) - ut
        avg_regret += regret
        if not reparam:
            gradient += regret * np.array(
                [
                    digamma(prior_alphas + prior_betas) - digamma(prior_alphas) + np.log(bandit[0]),
                    digamma(prior_alphas + prior_betas)
                    - digamma(prior_betas)
                    + np.log(1 - bandit[0]),
                ]
            )
        else:
            gradient += regret * np.array(
                [
                    (prior_alphas+prior_betas)*(-digamma(prior_alphas) + digamma(prior_betas) + np.log(bandit[0]/(1-bandit[0]))),
                    prior_alphas/(prior_alphas+prior_betas)*(-digamma(prior_alphas) + digamma(prior_betas) + np.log(bandit[0]/(1-bandit[0])))
                    + digamma(prior_alphas + prior_betas)
                    - digamma(prior_betas)+np.log(1 - bandit[0])
                ])
    return gradient / n_rollouts, avg_bayes_opt_ut / n_rollouts, regret / n_rollouts


def tangent(epsilon, belief):
    N = 20
    # nrollouts should depend on epsiloon
    num_actions = 20  # Horizon
    print(belief)
    gittins = bmab_gi_multiple_ab(belief[0], belief[1], gamma, N, num_actions, tol=5e-5)
    gradient, _, _ = bmab_bayes_opt_ut_via_rollouts(
        gittins, belief[0], belief[1], gamma=gamma, n_rollouts=n_rollouts
    )
    print(gradient)
    return gradient


out = cutting_plane(dim, constraints, b_ub, epsilon, delta, tangent, 30, 1000)

plot_all(out["constraints"], out["b_ub"], limits="beta")
