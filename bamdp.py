# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:38:33 2022

@author: grover
"""

class BAMDP: #Doesn't contain sampling methods. Kept seperate and called during Tree expansion
    def __init__(self,states,actions,rewards,transitions,mdpList,disc=0.99, horizon=None):
        self.gamma = disc
        if horizon is None:
            self.horizon = int(1.0/(1-disc))
        else:
            self.horizon = horizon
        self.states = states
        self.actions = actions
        self.rewards = rewards #Assuming a Dict of dict, r[mdp] = {sa: }
        self.transitions = transitions #Dict^3 [mdp][sa] = {s1,s2...sn} #again dictionary
        self.mdpList = mdpList
        #self.marginal[sa]       #Cant save cause changes with changing belief

    
    def getMarginalR(self,belief,sa=None,mdpList=None): #Sometimes cheaper to compute for a single sa
        
        #if sa and sa in self.marginal:
        #    return self.marginal[sa]       #Cant save cause changes with changing belief
        if mdpList == None:
            mdpList = self.mdpList
            
        marginalR = {}
        
        expectedReward = 0.0
        if sa:
            for mdp in mdpList:
                expectedReward += self.rewards[mdp][sa] * belief[mdp] #Assume belief to be dict
            marginalR[sa] = expectedReward
            
    
        else: #Compute for whole BAMDP once
            for s in self.states:
                for a in self.actions:
                    expectedReward = 0.0
                    for mdp in mdpList:
                        expectedReward += self.rewards[mdp][sa] * belief[mdp]
                        marginalR[s,a] = expectedReward
        
        return marginalR
    
    def getMarginalP(self,belief,sa=None,mdpList=None):

        if mdpList == None:
            mdpList = self.mdpList
        
        marginalP = {}
        
        
        if sa:
            marginal = {}
            for snext in self.states:
                marginal_snext  = 0.0
                for mdp in mdpList:
                    marginal_snext += self.transitions[mdp][sa][snext] * belief[mdp]
                marginal[snext] = marginal_snext
                    
            marginalP[sa] = marginal

        else:
            for s in self.states:
                for a in self.actions:
                    marginal = {}
                    for snext in self.states:
                        marginal_snext  = 0.0
                        for mdp in mdpList:
                            marginal_snext += self.transitions[mdp][sa][snext] * belief[mdp]
                            marginal[snext] = marginal_snext
                    marginalP[s,a] = marginal
        
#        return marginalP

        #More efficient to put Bayes rule here
        posteriors = {}
        for snext in self.states:
            marginal = marginalP[sa][snext]
            posterior = {}
            for mdp in mdpList:
                posterior[mdp] = 0.0
                if marginal:
                    posterior[mdp] =  self.transitions[mdp][sa][snext] * belief[mdp] / marginal
            posteriors[snext] = posterior

        return marginalP,posteriors

    #Bayes rule gives New Belief
    def getPosterior(self,belief,sa,snext,mdpList=None): #Didnt implement for all sa
        if mdpList == None:
            mdpList = self.mdpList
        
        posterior = {}
        marginalP , _ = self.getMarginalP(belief,sa,mdpList)
        marginal = marginalP[sa][snext]
        for mdp in mdpList:
            posterior[mdp] = 0.0
            if marginal:
                posterior[mdp] =  self.transitions[mdp][sa][snext] * belief[mdp] / marginal
        
        return posterior
        
        

#https://stackoverflow.com/questions/4901815/object-of-custom-type-as-dictionary-key
# ^^ States and Actions can be arbitary objects