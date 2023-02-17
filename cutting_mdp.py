from cuttingplane import cutting_plane, get_initial_constraints, plot_all
import numpy as np
from TreeSearchBAMDP import Node, TreeSearch
from bamdp import BAMDP
from environments import get_environment
import copy

from MDPUtils import valueIteration



def tangent(epsilon, beta, bv, mdpList, bamdp, state_id):
    beta = np.append(beta, 1-np.sum(beta))
    assert np.sum(beta)>0.9999 and np.sum(beta)<1.00001
    assert np.max(beta)<= 1.001 and np.min(beta)>= -0.001
    node = Node(bamdp, state_id, dict(enumerate(beta)))
    tree = TreeSearch(bamdp, node)
    tree.search()
    tree.Eval_VI()
    # print("BV", bv)

    print("Value", node.otherdata['V'])
    print("Regret: ", bv@beta-node.otherdata['V'])



    v_grad = tree.get_gradient(node)
    v_grad = np.array([v_grad[i] for i in mdpList])

    r_grad = bv - v_grad
    # Transform to regret
    print("Grad", r_grad)

    # belief sums to one, need to adjust gradient.
    r_grad[:-1] -= r_grad[-1]

    #Ensure maximization
    transform_grad = -r_grad[:-1]

    print("transformGrad", transform_grad)
    return transform_grad


def get_bayes_optimal_value(bamdp, mdpList, state_id):


    bv = []
    for i in mdpList:
        print(i)
        beta = np.zeros(len(mdpList))
        beta[i] = 1
        node = Node(bamdp, state_id, dict(enumerate(beta)))
        tree = TreeSearch(bamdp, node)
        tree.search()
        tree.Eval_VI()
        bv.append(node.otherdata['V'])
        #bv.append(tree.get_gradient(node)[i])

    return np.array(bv)


def main(dim):
    seed = 567
    dim, states, actions, mdpList, rewards, transitions = get_environment("random", dim=dim, seed=seed)
    print("Seed", seed)
    epsilon = 0
    delta = 0
    gamma = 1
    discount = gamma
    H = 5
    state_id = 0

    constraints, b_ub = get_initial_constraints(dim)

    bamdp = BAMDP(range(states), range(actions), rewards, transitions, mdpList, discount, horizon=H)
    bayes_value = get_bayes_optimal_value(bamdp, mdpList, state_id)
    print("Bayes_value: ", bayes_value )

    regret_pair = np.zeros((dim+1, dim+1))

    for i in range(dim+1):
        b = np.zeros(dim+1)
        b[i] = 1

        node = Node(bamdp, state_id, dict(enumerate(b)))
        tree = TreeSearch(bamdp, node)
        tree.search()
        tree.Eval_VI()
        grad = tree.get_gradient(node)
        grad = np.array([grad[i] for i in mdpList])
        #grad -= bayes_value
        regret_pair[i,:] = bayes_value - grad
    print(regret_pair)

    b = np.ones(dim + 1)
    b=b/np.sum(b)

    node = Node(bamdp, state_id, dict(enumerate(b)))
    tree = TreeSearch(bamdp, node)
    tree.search()
    tree.Eval_VI()
    grad = tree.get_gradient(node)
    grad = np.array([grad[i] for i in mdpList])
    print("uniform bel regret", bayes_value-grad)

    out = cutting_plane(dim, constraints, b_ub, epsilon, delta, lambda epsilon, beta: tangent(epsilon, beta,
                                                                                              bayes_value,
                                                                                              mdpList, bamdp,
                                                                                              state_id), 30, 10000,
                                                                                                plot=False)

    print(out)
    plot_all(out["constraints"], out["b_ub"], limits="none")
    out["transitions"] = transitions
    out["rewards"] = rewards
    out["dim"] = dim
    out["states"] = states
    out["actions"] = actions
    out["horizon"] = H
    out["gamma"] =discount
    out["epsilon"] = epsilon
    out["bv"] = bayes_value
    out["bamdp"] = bamdp
    out["state_id"] = state_id
    out["regret_pair"] = regret_pair

    np.save("out_nogamma{}.npy".format(dim),out)

if __name__ == "__main__":
    print("dim 1")
    main(dim=1)
    print("dim 2")
    main(dim=2)

