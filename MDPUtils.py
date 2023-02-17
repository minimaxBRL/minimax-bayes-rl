# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:21:06 2022

@author: grover
"""

def valueIteration(bamdp,targetmdp): #particular mdp on which to perform VI
    states = bamdp.states
    actions = bamdp.actions
    horizon,gamma = bamdp.horizon,bamdp.gamma
    reward = bamdp.rewards[targetmdp]
    transition = bamdp.transitions[targetmdp]
    SA = [(s,a) for a in actions for s in states]
    condition = True

    V = {}
    for s in states:
        V[s] = 0.0
    policy = {}

    Q = {}
    while condition:
        for sa in SA:
            Q[sa] = 0
            for sn in states:
                Q[sa] += transition[sa][sn] * (reward[sa] + gamma * V[sn])
        
        #Computing value and policy
        max_diff = 0.0
        for s in states:
            lis = []
            for a in actions:
                lis.append((Q[s,a],a))
            max_element = max(lis,key=lambda x: x[0])
            
            if max_element[0] - V[s] > max_diff:
                max_diff = max_element[0] - V[s]
            
            V[s] = max_element[0]
            policy[s] = max_element[1]
    
        if max_diff < 1e-6:
            condition = False
            
    return V,policy,Q
