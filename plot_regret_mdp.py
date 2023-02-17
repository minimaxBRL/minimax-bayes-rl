import numpy as np
from TreeSearchBAMDP import Node, TreeSearch
from bamdp import BAMDP
import matplotlib.pyplot as plt









def get_psrl_plane(beta,regret_pair):
    regret_grad = np.sum(regret_pair*beta[:,None], axis=0) #Expected value over policies
    regret = regret_grad@beta
    r_grad = regret_grad
    r_grad[:-1] -= r_grad[-1]
    transform_grad = r_grad[:-1]

    return regret, r_grad, transform_grad




if __name__ == "__main__":
    plt.rcParams["savefig.format"] = "pdf"

    for file in ["out_nogamma1.npy", "out_nogamma2.npy"]:
        out = np.load(file, allow_pickle=True).item()

        transitions = out["transitions"]
        rewards = out["rewards"]
        dim = out["dim"]
        states = out["states"]
        actions = out["actions"]
        H = out["horizon"]
        discount = out["gamma"]
        epsilon = out["epsilon"]
        bv = out["bv"]
        bamdp = out["bamdp"]
        state_id = out["state_id"]
        mdpList = bamdp.mdpList
        regret_pair = out["regret_pair"]

        minmaxbel = out["belief"]


        def get_plane(beta):
            assert len(beta) == dim + 1
            node = Node(bamdp, state_id, dict(enumerate(beta)))
            tree = TreeSearch(bamdp, node)
            tree.search()
            tree.Eval_VI()
            regret = bv @ beta - node.otherdata['V']

            v_grad = tree.get_gradient(node)
            v_grad = np.array([v_grad[i] for i in mdpList])

            r_grad = bv - v_grad
            r_grad[:-1] -= r_grad[-1]
            transform_grad = r_grad[:-1]

            return regret, r_grad, transform_grad

        #Get right policy
        eps = 0.001
        mm_regret, mm_grad, mm_transform_grad = get_plane(minmaxbel)


        mm_u_regret, mm_u_grad, mm_u_transform_grad = get_plane(minmaxbel+eps)


        #Get left policy
        mm_l_regret, mm_l_grad, mm_l_transform_grad = get_plane(minmaxbel-eps)
        # uniform belief
        uniform_belief = np.ones(dim + 1)
        uniform_belief /= np.sum(uniform_belief)
        uniform_regret, uniform_grad, uniform_transform_grad = get_plane(uniform_belief)

        if dim == 1:
            bel = np.linspace(0, 1)

            plt.ylabel("Bayesian Regret")
            plt.xlabel(r'$P(\mu_1)$')
            plt.plot(bel, mm_u_transform_grad * (bel - minmaxbel[0]) + mm_regret, label="Best response", linestyle='dotted')
            plt.plot(bel, mm_l_transform_grad * (bel - minmaxbel[0])
                    + mm_regret, label="Best response", linestyle='dotted')

            # This policy can always be found with probability p selecting policy 1 and 1-p for policy 2
            #  such that k_1*p + k_2*(1-p) = 0
            plt.plot(bel, mm_regret*np.ones_like(bel), label="Minimax", linestyle='dotted')
            

            #plt.plot(bel, uniform_transform_grad * (bel - uniform_belief[0]) + uniform_regret, label="Uniform policy")
            if False:
                n_psrl = 5
                for psrl_beta in range(0,n_psrl+1):
                    psrl_beta/=n_psrl
                    beta = np.array([psrl_beta, 1-psrl_beta])
                    psrl_regret, _, grad = get_psrl_plane(beta, regret_pair)
                    plt.plot(bel, grad*(bel-psrl_beta) + psrl_regret,
                            label=r"PSRL $\beta={}$".format(psrl_beta))




            n_psrl = 100
            regrets = []
            bels = []
            for psrl_beta in range(0, n_psrl + 1):
                psrl_beta /= n_psrl
                beta = np.array([psrl_beta, 1 - psrl_beta])
                regret, _, grad = get_plane(beta)
                regrets.append(regret)
                bels.append(psrl_beta)

            plt.plot(bels, regrets, label=r"Bayes optimal")
            plt.legend()

            plt.savefig("../tex/img/{}_lineplot".format(file.split(".")[0]))


            n_psrl = 100
            regrets = []
            bels = []
            for psrl_beta in range(0,n_psrl+1):
                psrl_beta/=n_psrl
                beta = np.array([psrl_beta, 1-psrl_beta])
                psrl_regret, _, grad = get_psrl_plane(beta, regret_pair)
                regrets.append(psrl_regret)
                bels.append(psrl_beta)

            plt.plot(bels, regrets, label=r"PSRL")
            plt.legend()
            plt.savefig("../tex/img/{}_psrl_lineplot".format(file.split(".")[0]))
        plt.close()
