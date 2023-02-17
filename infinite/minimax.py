import torch
import numpy as np

# expects torch tensors
def rollout_adaptive(T, R, policy, max_steps=1000, state_prior=None, gamma=0.99, stationary=False):

    nS = T.shape[0]
    nA = T.shape[1]
    ohe = torch.eye(nS)

    if state_prior is None:
        state_prior = torch.distributions.Categorical(torch.ones(nS) / nS)


    s = state_prior.sample().detach().item()
    history = torch.zeros((nS, nA, nS))
    reward_trajectory = torch.zeros((max_steps))
    log_prob_trajectory = torch.zeros((max_steps))
    log_prob_s_trajectory = torch.zeros((max_steps))

    for t in range(max_steps):

        if stationary:
            concat_state = torch.cat((torch.zeros((nS * nA * nS)), ohe[s]))
        else:
            concat_state = torch.cat((history.flatten(), ohe[s]))
        action_probs = policy(concat_state)
        m1 = torch.distributions.Categorical(action_probs)
        a = m1.sample()

        m2 = torch.distributions.Categorical(probs=T[s, a])
        s_ = m2.sample()

        r = R[s, a]

        reward_trajectory[t] = r
        log_prob_trajectory[t] = m1.log_prob(a)
        log_prob_s_trajectory[t] = m2.log_prob(s_)
        history[s, a, s_] += 1.0 / max_steps
        s = s_

    return reward_trajectory, log_prob_trajectory, log_prob_s_trajectory

def rollout_optimal(T, R, policy, max_steps=1000, state_prior=None):

    nS = T.shape[0]
    nA = T.shape[1]

    if state_prior is None:
        state_prior = torch.distributions.Categorical(torch.ones(nS) / nS)


    s = state_prior.sample().detach().item()
    reward_trajectory = torch.zeros((max_steps))
    log_prob_s_trajectory = torch.zeros((max_steps))

    for t in range(max_steps):

        a = np.random.choice(range(nA), p=policy[s])

        m2 = torch.distributions.Categorical(probs=T[s, a])
        s_ = m2.sample()

        r = R[s, a]

        reward_trajectory[t] = r
        log_prob_s_trajectory[t] = m2.log_prob(s_)
        s = s_

    return reward_trajectory, log_prob_s_trajectory

# expects torch tensors
def collect_rollouts(T_prior, R_prior, policy, max_steps, state_prior=None, gamma=0.99, num_rollouts=10, num_mdps=10, stationary=False):

    complete_reward_traj = torch.zeros((num_mdps, num_rollouts, max_steps))
    complete_opt_traj = torch.zeros((num_mdps, num_rollouts, max_steps))
    complete_log_prob_traj = torch.zeros((num_mdps, num_rollouts, max_steps))
    complete_log_prob_s_traj = torch.zeros((num_mdps, num_rollouts, max_steps))
    complete_log_prob_opt_s_traj = torch.zeros((num_mdps, num_rollouts, max_steps))


    nS = T_prior.shape[0]
    nA = T_prior.shape[1]

    if state_prior is None:
        state_prior = torch.distributions.Categorical(torch.ones(nS) / nS)

    for i in range(num_mdps):

        T = torch.distributions.Dirichlet(concentration=T_prior).rsample()
        R = R_prior.sample()
        _, _, pol = compute_optimal_policy(T, R, gamma=gamma)

        for j in range(num_rollouts):
            rs, lg, lgs = rollout_adaptive(T, R, policy, max_steps, state_prior, stationary=stationary)
            opt_rs, opt_lgs = rollout_optimal(T, R, pol, max_steps, state_prior)
            complete_reward_traj[i, j, :] = rs
            complete_opt_traj[i, j, :] = opt_rs
            complete_log_prob_traj[i, j, :] = lg
            complete_log_prob_s_traj[i, j, :] = lgs
            complete_log_prob_opt_s_traj[i, j, :] = opt_lgs

    return complete_reward_traj, complete_log_prob_traj, complete_log_prob_s_traj, complete_opt_traj, complete_log_prob_opt_s_traj

def compute_optimal_policy(T, R, gamma=0.99, eps=1e-7):


    num_states = R.shape[0]
    num_actions = R.shape[1]

    converged = False

    V = torch.zeros((num_states))
    Q = torch.zeros((num_states, num_actions))

    while not converged:
        Q = R + gamma * T @ V
        V_new = torch.max(Q, axis=1)[0]

        if torch.linalg.norm(V_new-V) <= eps:
            converged = True

        V = V_new

    idx = torch.argmax(Q, axis=1)
    policy = torch.zeros((num_states, num_actions))
    for s in range(num_states):
        policy[s][idx[s]] = 1.0

    return V, Q, policy

class NNPolicy(torch.nn.Module):
    def __init__(self, nS, nA):
        super(NNPolicy, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.hidden_size = 100
        self.fc1 = torch.nn.Linear(nS * nA * nS + nS, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = torch.nn.Linear(nS * nA * nS + nS, nA)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(self.hidden_size, nA)

    def forward(self, x):

        #x = x.float()

        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return torch.nn.functional.softmax(x, dim=0)

import time
def run_experiment(num_iterations=1000, type='both', stationary=False, prior=None, _policy=None):

    nS = 5
    nA = 2

    from envs import Chain
    env = Chain()
    lr_w = 2e-4
    lr_b = 1e-4

    if prior is None:
        T_prior_params = torch.ones((nS, nA, nS), requires_grad=True) # random transition matrix
    else:
        T_prior_params = torch.load(prior)

    if _policy is None:
        policy = NNPolicy(nS, nA)

    else:
        policy = NNPolicy(nS, nA)
        policy.load_state_dict(torch.load(_policy))
        policy.train()

    R_prior = torch.distributions.Normal(loc=torch.tensor(env.R, dtype=torch.float32), scale=torch.ones((nS, nA))/1000.0) # chain reward function

    optimizer_b = torch.optim.SGD([T_prior_params], lr=lr_b)

    if type == 'both':
        optimizer = torch.optim.SGD(policy.parameters(), lr=lr_w)

    else:
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr_w)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_iterations//10, gamma=0.75)
    scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=num_iterations // 10, gamma=0.75)
    start_time = time.time()

    num_mdps = 18
    num_rollouts = 2
    max_steps = 1000

    it_regret = 0
    it_opt = 0
    it_val = 0

    init_w = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
    init_b = T_prior_params.detach()

    from torch.utils.tensorboard import SummaryWriter

    if stationary:
        outdir = 'runs/' + type + "_stationary/"
    else:
        outdir = 'runs/' + type + "_adaptive/"

    writer = SummaryWriter(log_dir=outdir)

    for t in range(num_iterations):
        if t % (num_iterations // 100) == 0:
            with torch.no_grad():
                _V, _R, _WC, _25 = eval_policy_prior(T_prior_params, R_prior, policy, max_steps=max_steps, stationary=stationary)
                writer.add_scalar(tag='Eval/Avg. Value', scalar_value=_V.detach().item(), global_step=t)
                writer.add_scalar(tag='Eval/Avg. Regret', scalar_value=_R.detach().item(), global_step=t)
                writer.add_scalar(tag='Eval/WC. Regret', scalar_value=_WC.detach().item(), global_step=t)
                writer.add_scalar(tag='Eval/Low 25. Regret', scalar_value=_25.detach().item(), global_step=t)
                for s_ in range(nS):
                    ohe = torch.eye(nS)
                    history = torch.ones(nS, nA, nS) / max_steps
                    concat_state = torch.cat((history.flatten(), ohe[s_]))
                    action_probs = policy(concat_state)
                    writer.add_scalar(tag='Eval/Action Probs a='+str(s_), scalar_value=action_probs[0].item(), global_step=t)
            print(t, "Eval. ", _V.detach().item(), _R.detach().item(), _25.detach().item(), _WC.detach().item())

        crs, clg, clgs, ors, optlgs = collect_rollouts(T_prior_params, R_prior, policy, max_steps=max_steps, num_rollouts=num_rollouts, num_mdps=num_mdps, stationary=stationary)

        loss_w = torch.tensor(0.0)
        loss_b = torch.tensor(0.0)

        regret = torch.mean(torch.sum(ors[:, :, :], axis=2), axis=1)-torch.mean(torch.sum(crs[:, :, :], axis=2), axis=1)

        opt_val = torch.mean(torch.sum(ors[:, :, :], axis=2), axis=1)
        val = torch.mean(torch.sum(crs[:, :, :], axis=2), axis=1)

        for i in range(num_mdps):
            for j in range(num_rollouts):
                bl = torch.mean(crs[i, j])
                bl_opt = torch.mean(ors[i, j])
                for k in range(max_steps):
                    loss_w += -clg[i, j, k] * (torch.sum(crs[i, j, k:])-(max_steps-k)*bl)
                    loss_b -= optlgs[i, j, k] * (torch.sum(ors[i, j, k:])-(max_steps-k)*bl_opt) - \
                              clgs[i, j, k]   * (torch.sum(crs[i, j, k:])-(max_steps-k)*bl)  # maximises regret

        loss_w /= num_mdps * num_rollouts
        loss_b /= num_mdps * num_rollouts

        if type == 'policy':
            loss_b = torch.tensor(0.0)
        elif type == 'prior':
            loss_w = torch.tensor(0.0)
        else:
            pass

        optimizer.zero_grad()
        optimizer_b.zero_grad()

        if type == 'both':

            loss_b.backward(retain_graph=True)
            loss_w.backward(retain_graph=False)
            optimizer.step()
            optimizer_b.step()
            scheduler.step()
            scheduler_b.step()

        elif type == 'prior':
            loss_b.backward()
            optimizer_b.step()
            scheduler_b.step()

        elif type == 'policy':
            loss_w.backward()
            optimizer.step()
            scheduler.step()

        b_grad = torch.tensor(0.0)
        w_grad = torch.tensor(0.0)
        if not T_prior_params.grad is None:
            b_grad = torch.sum(T_prior_params.grad)
        for param in policy.parameters():
            if not param.grad is None:
                w_grad += torch.sum(param.grad)
        T_prior_params.data = torch.clamp(T_prior_params.data, min=1e-2, max=None)

        current_w = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
        dist_w = torch.linalg.norm(init_w-current_w)
        dist_b = torch.linalg.norm(init_b-T_prior_params.detach())

        it_regret = (it_regret * t + (torch.mean(opt_val - val))) / (t+1)
        it_opt = (it_opt * t + (torch.mean(opt_val))) / (t+1)
        it_val = (it_val * t + (torch.mean(val))) / (t+1)

        writer.add_scalar(tag='Metrics/Avg. Value', scalar_value=torch.mean(torch.sum(crs[:, :, :], axis=2), axis=(0, 1)).item(),
                          global_step=t)
        writer.add_scalar(tag='Metrics/Avg. Regret', scalar_value=torch.mean(regret).item(),
                          global_step=t)
        writer.add_scalar(tag='Metrics/Running avg. Regret', scalar_value=it_regret.item(),
                          global_step=t)
        writer.add_scalar(tag='Metrics/Running avg. Value', scalar_value=it_val.item(),
                          global_step=t)
        writer.add_scalar(tag='Metrics/Running avg. Optimal Value', scalar_value=it_opt.item(),
                          global_step=t)
        writer.add_scalar(tag='Loss/Policy Loss', scalar_value=loss_w.item(),
                          global_step=t)
        writer.add_scalar(tag='Loss/Prior Loss', scalar_value=loss_b.item(),
                          global_step=t)
        writer.add_scalar(tag='Gradients/Policy Gradient', scalar_value=w_grad.item(),
                          global_step=t)
        writer.add_scalar(tag='Gradients/Prior Gradient', scalar_value=b_grad.item(),
                          global_step=t)
        writer.add_scalar(tag='Policy/InitNorm', scalar_value=dist_w.item(),
                          global_step=t)
        writer.add_scalar(tag='Prior/InitNorm', scalar_value=dist_b.item(),
                          global_step=t)




        print(t, "Train. ", time.time()-start_time, torch.mean(torch.sum(crs[:, :, :], axis=2), axis=(0, 1)).item(),
              torch.mean(regret).item(), it_regret.item(), w_grad.item(), b_grad.item(), dist_w.item(), dist_b.item(), loss_w.item(), loss_b.item())

        if t % (num_iterations // 1000) == 0:
            torch.save(policy.state_dict(), outdir + str(seed) + "_policy_weights.npy")
            torch.save(T_prior_params, outdir + str(seed) + "_prior_weights.npy")

        del regret
        del val
        del opt_val
        del crs
        del clg
        del clgs
        del ors
        del optlgs
        del loss_w
        del loss_b

def eval_policy_prior(T_prior, R_prior, policy, max_steps=1000, state_prior=None, gamma=0.99, num_rollouts=4, num_mdps=25, stationary=False):

    with torch.no_grad():
        policy.eval()
        crs, _, _, ors, _ = collect_rollouts(T_prior, R_prior, policy, state_prior=state_prior, gamma=gamma, max_steps=max_steps,
                                               num_rollouts=num_rollouts, num_mdps=num_mdps, stationary=stationary)
        policy.train()
        regrets = torch.mean(torch.sum(ors, axis=2)-torch.sum(crs, axis=2), axis=(0, 1))
        avg_val = torch.mean(torch.sum(crs, axis=2), axis=(0, 1))
        wc_regret = torch.max(torch.mean(torch.sum(ors, axis=2), axis=1)-torch.mean(torch.sum(crs, axis=2), axis=1))
        sort, _ = torch.sort(torch.mean(torch.sum(ors, axis=2), axis=1) - torch.mean(torch.sum(crs, axis=2), axis=1),
                   descending=True)
        low_25 = torch.mean(sort[0:(num_mdps//4)])
    return avg_val, regrets, wc_regret, low_25

# evaluates all the policies on all the priors
def eval_policies_priors(T_priors, R_prior, policies, max_steps=1000, state_prior=None, gamma=0.99, num_rollouts=20, num_mdps=1000, stationary=False):

    with torch.no_grad():

        num_policies = len(policies)
        num_priors = len(T_priors)
        collected_rewards = torch.zeros((num_priors, num_policies, num_mdps, num_rollouts, max_steps))
        collected_opt_rewards = torch.zeros((num_priors, num_mdps, num_rollouts, max_steps))
        nS = T_priors[0].shape[0]
        state_priors = torch.eye(nS)

        R = R_prior.sample()

        for i in range(len(T_priors)):
            active_prior = T_priors[i]
            Ts = torch.distributions.Dirichlet(concentration=active_prior).rsample(sample_shape=torch.Size((num_mdps,)))

            for j in range(num_mdps):

                _, _, pol = compute_optimal_policy(Ts[j], R, gamma=gamma)

                for k in range(num_rollouts):
                    res = rollout_optimal(Ts[j], R, pol, max_steps, torch.distributions.Categorical(state_priors[k % nS]))
                    collected_opt_rewards[i, j, k, :] = res[0]

                for k in range(len(policies)):
                    for l in range(num_rollouts):
                        rs, _, _ = rollout_adaptive(Ts[j], R, policies[k], max_steps, torch.distributions.Categorical(state_priors[k % nS]), stationary=stationary)
                        collected_rewards[i, k, j, l, :] = rs

        values = torch.mean(torch.sum(collected_rewards, axis=4), axis=3)
        opt_values = torch.mean(torch.sum(collected_opt_rewards, axis=3), axis=2)
        opt_values = opt_values.unsqueeze(1).repeat(1, len(policies), 1)

        return opt_values-values

def final_eval(num_rollouts, num_mdps,
               maximin_prior_path,
               minimax_policy_path,
               br_uniform_policy_path,
               br_maximin_policy_path):

    priors = []
    policies = []

    nS = 5
    nA = 2

    if num_rollouts is None:
        num_rollouts = 10
    if num_mdps is None:
        num_mdps = 1000
    if maximin_prior_path is None:
        maximin_prior_path = "params/maximin_prior.npy"
    if minimax_policy_path is None:
        minimax_policy_path = "params/minimax_policy.npy"
    if br_uniform_policy_path is None:
        br_uniform_policy_path = "params/br_uniform_policy.npy"
    if br_maximin_policy_path is None:
        br_maximin_policy_path = "params/br_maximin_policy.npy"

    from envs import Chain
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    env = Chain()

    state_prior = torch.distributions.Categorical(torch.ones(nS) / nS)

    uniform_prior = torch.ones((nS, nA, nS)) # random transition matrix
    chain_prior = torch.tensor(torch.tensor(env.T, dtype=torch.float32)*1000.0 + torch.ones((nS, nA, nS)) / 100.0) # chain transition matrix
    maximin_prior = torch.load(maximin_prior_path)

    priors.append(uniform_prior)
    priors.append((2/3 * uniform_prior + 1/3 * maximin_prior))
    priors.append((1/3 * uniform_prior + 2/3 * maximin_prior))
    priors.append(maximin_prior)
    priors.append(uniform_prior / 1000.0)
    priors.append(chain_prior)

    minimax_policy = NNPolicy(nS, nA)
    minimax_policy.load_state_dict(torch.load(minimax_policy_path))
    minimax_policy.eval()

    uniform_policy = NNPolicy(nS, nA)
    uniform_policy.load_state_dict(torch.load(br_uniform_policy_path))
    uniform_policy.eval()

    br_maximin_policy = NNPolicy(nS, nA)
    br_maximin_policy.load_state_dict(torch.load(br_maximin_policy_path))
    br_maximin_policy.eval()

    policies.append(uniform_policy)
    policies.append(minimax_policy)
    policies.append(br_maximin_policy)

    R_prior = torch.distributions.Normal(loc=torch.tensor(env.R, dtype=torch.float32), scale=torch.ones((nS, nA))/1000.0) # chain reward function

    regrets = eval_policies_priors(priors, R_prior, policies, max_steps=1000, state_prior=state_prior,
                         gamma=0.99, num_rollouts=num_rollouts, num_mdps=num_mdps, stationary=False)
    regrets = regrets.detach().numpy()

    policy_str = [r'$\pi^{*}(\beta^{1})$', r'$\pi^{*}$',  r'$\pi^{*}(\beta^{*})$']
    prior_str = [r'$\beta^{1}$', " ", "  ", r'$\beta^{*}$', r'$\beta^{D}$', r'$\beta^{Chain}$']

    dfs = []

    for i in range(regrets.shape[0]):
        for j in range(regrets.shape[1]):
            regret_df = pd.DataFrame(regrets[i, j, :], columns=["Regret"])
            regret_df["Prior"] = [prior_str[i]] * regrets.shape[2]
            regret_df["Policy"] = [policy_str[j]] * regrets.shape[2]
            dfs.append(regret_df)
    df = pd.concat(dfs)

    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(x="Prior", y="Regret",
                hue="Policy", palette=["m", "g", "b"],
                data=df, showfliers = False, whis=[25, 99.9], showmeans=True, meanline=True,
                meanprops=dict(linestyle='--', color=list([111/255, 111/255, 111/255])))

    sns.despine(offset=10, trim=True)
    plt.savefig("boxplot.pdf", bbox_inches='tight')


def main():

    final_eval()
    #fwagagw()
    #prior = torch.load("runs/both_adaptive/1672213957_prior_weights.npy")
    #print(prior)
    #prior = torch.load("results/params/maximin_prior.npy")
    #print(prior)
    #dwagaga()
    #run_experiment(1000, 'policy', stationary=True)
    #run_experiment(1000, 'prior', _policy="parameters/new/br_uniform_policy.npy")



    # 1000 iterations @ 18 mdps, 2 rollouts ~approx 24 hours
    #run_experiment(10000, 'policy', seed=100) # best response to uniform prior
    #run_experiment(10000, 'both', seed=150) # minimax solution
    #run_experiment(10000, 'policy', prior="results/params/new/maximin_prior.npy", seed=150) # best response policy to maximin prior
    #run_experiment(1000, 'prior', _policy="results/params/br_uniform_policy.npy") # best response prior to minimax policy
    #run_experiment(1000, 'prior', _policy="parameters/br_uniform_policy.npy")  # best response prior to best response policy to uniform prior

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--plot', help="Whether to print or train", action='store_true')
    parser.add_argument('--mmpol', help="Path to minimax policy", type=str)
    parser.add_argument('--mmpri', help="Path to minimax prior", type=str)
    parser.add_argument('--unpol', help="Path to uniform policy", type=str)
    parser.add_argument('--brpol', help="Path to best response to minimax policy", type=str)
    parser.add_argument('--num_rollouts', help="Number of rollouts >= 2", type=int)
    parser.add_argument('--num_mdps', help="Number of MDPs >= 4", type=int)
    parser.add_argument('--seed', help="Seed", type=int)
    parser.add_argument('--num_iterations', help="Number of gradient steps (iterations)", type=int)
    parser.add_argument('--both', help="Optimise both policy and prior?", action='store_true')
    parser.add_argument('--policy', help="Optimise policy? Ignored if both is set.", action='store_true')
    parser.add_argument('--prior', help="Optimise prior? Ignored if both is set.", action='store_true')
    parser.add_argument('--train_prior', help='Prior to start optimise from, uniform if not set.', type=str)
    parser.add_argument('--train_policy', help='Policy to start optimise from, uniform if not set.', type=str)

    args = parser.parse_args()

    if args.seed is None:

        torch.manual_seed(42)
        np.random.seed(42)

    else:

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # To plot or train?
    if args.plot:
        final_eval(num_rollouts=args.num_rollouts,
                   num_mdps=args.num_mdps,
                   maximin_prior_path=args.mmpri,
                   minimax_policy_path=args.mmpol,
                   br_uniform_policy_path=args.unpol,
                   br_maximin_policy_path=args.brpol)
    else:
        its = args.num_iterations
        if args.num_iterations is None:
            its = 1000

        if args.both or (args.prior and args.policy):
            run_experiment(num_iterations=its, type='both', stationary=False, prior=args.train_prior, _policy=args.train_policy)

        elif args.prior:
            run_experiment(num_iterations=its, type='prior', stationary=False, prior=args.train_prior, _policy=args.train_policy)

        elif args.policy:
            run_experiment(num_iterations=its, type='policy', stationary=False, prior=args.train_prior,_policy=args.train_policy)

        else:
            pass
