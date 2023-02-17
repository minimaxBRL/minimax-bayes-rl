from cuttingplane import cutting_plane, get_initial_constraints, plot_all
import numpy as np
from TreeSearchBAMDP import Node, TreeSearch
from bamdp import BAMDP
from environments import get_environment
import matplotlib.pyplot as plt
from cutting_mdp import get_bayes_optimal_value
import glob
import copy

from MDPUtils import valueIteration



def tangent(beta, bv, mdpList, bamdp, state_id):
    beta = np.append(beta, 1-np.sum(beta))
    if not (np.sum(beta)>0.9999 and np.sum(beta)<1.00001):
        print(beta)
    if not (np.max(beta)<= 1.001 and np.min(beta)>= -0.001):
        print(beta)
    node = Node(bamdp, state_id, dict(enumerate(beta)))
    tree = TreeSearch(bamdp, node)
    tree.search()
    tree.Eval_VI()
    regret = bv@beta-node.otherdata['V']
    v_grad = tree.get_gradient(node)
    v_grad = np.array([v_grad[i] for i in mdpList])
    #grad -= bv
    r_grad = bv - v_grad
    # Transform to regret

    # belief sums to one, need to adjust gradient.
    r_grad[:-1] -= r_grad[-1]

    transform_grad = r_grad[:-1]

    #print("transformGrad", transform_grad)
    return transform_grad, regret


def generate_data(filename):
    out = np.load("{}.npy".format(filename), allow_pickle=True).item()

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

    constraints, b_ub = get_initial_constraints(dim)

    bamdp = BAMDP(range(states), range(actions), rewards, transitions, mdpList, discount, horizon=H)
    bayes_value = get_bayes_optimal_value(bamdp, mdpList, state_id)
    print("Bayes_value: ", bayes_value )

    regret_pair = np.zeros((dim+1, dim+1))
    resolution = 25

    if dim ==1:
        resolution2 = 1
    else:
        resolution2 = resolution

    R = np.zeros((resolution, resolution2))
    grads = np.zeros((dim, resolution, resolution2))
    for i in range(resolution):
        print(i)
        for j in range(resolution2):
            if dim == 1:
               beta = np.array(i/resolution+1/(2*resolution))
            else:
                beta = np.array([i/resolution+1/(2*resolution),j/resolution2+1/(2*resolution2)])

            if np.sum(beta)<=1.0001:
                grads[:, i, j], R[i,j] = tangent(beta,bayes_value, mdpList, bamdp, state_id)
    np.save("{}_regretgrad.npy".format(filename), {"regret": R, "grads":grads })


def plot(filename):
    xmin = 0
    ymin = 0
    xmax = 1
    ymax = 1

    data  = np.load("{}_regretgrad.npy".format(filename), allow_pickle=True).item()
    R = data["regret"]
    resolution = R.shape[0]
    resolution2 = R.shape[1]
    grads = data["grads"]
    if resolution2 == 1:
        ymax=0.1
    plt.imshow(R.T,  origin="lower", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    maxlength = 1/(3*resolution)


    for i in range(resolution):
        print(i)
        for j in range(resolution2):
            beta = np.array([i/resolution+1/(2*resolution),j/resolution2+1/(2*resolution2)])
            g = grads[:,i,j]/np.max(np.abs(grads))
            if resolution2 == 1:
                plt.arrow(beta[0], 0.05, maxlength*g[0], 0, width=0.003)
                plt.tick_params(
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,# ticks along the top edge are off
                    left=False,
                    labelleft=False)  # labels along the bottom edge are off
            else:
                if np.sum(beta) <= 1.0001:

                  plt.arrow(beta[0], beta[1], maxlength*g[0], maxlength*g[1], width=0.001)

    plt.savefig('../tex/img/{}_mdpgrid'.format(filename))
    plt.close()

if __name__ == "__main__":
    plt.rcParams["savefig.format"] = "pdf"

    for file in ["out1.npy","out2.npy"]:
        filename = file.split(".")[0]
        #generate_data(filename)
        plot(filename)

