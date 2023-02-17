# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:13:19 2022

@author: grover
"""

import numpy as np
from TreeSearchBAMDP import Node, TreeSearch
from bamdp import BAMDP
from environments import get_environment
from MDPUtils import valueIteration

np.random.seed(123)
dim, states, actions, mdpList, rewards, transitions = get_environment("random")
epsilon = 0.0
delta = 0.02
gamma = 0.9
discount = gamma
state_id = 0
H = 4


bamdp = BAMDP(range(states), range(actions), rewards, transitions, mdpList, discount, horizon=H)

for i in range(dim+1):
    beta = np.zeros(dim+1)
    print("MDP", i)
    beta[i] = 1
    assert np.sum(beta)==1
    node = Node(bamdp, state_id, dict(enumerate(beta)))
    tree = TreeSearch(bamdp, node)
    tree.search()
    tree.Eval_VI()
    bv = valueIteration(bamdp,i)[0][0]
    print("BV", bv)
    print(node.otherdata['V'])
    print("Regret: ", bv-node.otherdata['V'])



beta = np.zeros(dim+1)
beta[0] = 0.5
beta[1] = 0.5
node = Node(bamdp, state_id, dict(enumerate(beta)))
tree = TreeSearch(bamdp, node)
tree.search()
tree.Eval_VI()
v1 = valueIteration(bamdp,0)[0][0]
v2 = valueIteration(bamdp,1)[0][0]
v1 = valueIteration(bamdp,0)[0][0]
bv = beta[0]*v1+beta[1]*v2
print("BV", bv)
print(node.otherdata['V'])
print("Regret: ", bv-node.otherdata['V'])

