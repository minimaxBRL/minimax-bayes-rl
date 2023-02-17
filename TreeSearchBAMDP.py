# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:01:40 2022

@author: grover
"""
import random,collections
from bamdp import BAMDP
from MDPUtils import valueIteration

class Node:
  def __init__(self,bamdp,s,bel,level=0,parent=None,a=None,prob=None,r=None):
    self.bamdp = bamdp
    self.parent = parent
    self.depth = level
    self.s = s
    self.bel = bel
    self.prob = prob
    self.r = r
    self.act = a
    #self.myname = id(self)
    self.otherdata = {"Vmdp": collections.defaultdict(float)}


  #Full expansion, i.e. not removing low-prob children, while removing has same effect as sampling next nodes
  def Expand(self,full=True):
    lis_A = self.bamdp.actions
    if self.depth >= self.bamdp.horizon:
      return []

    children = []
    for a in lis_A:
      probs,posteriors = self.bamdp.getMarginalP(self.bel,(self.s,a))
      probs = probs[self.s,a]
      r = self.bamdp.getMarginalR(self.bel,(self.s,a)) [self.s,a]
      data = zip(self.bamdp.states,[probs[s] for s in self.bamdp.states],[r]*len(probs))
      cumsum_prob = 0.0
      for obs,prob,r in sorted(data,key=lambda x: x[1],reverse=True):
        cumsum_prob += prob
        new_bel = posteriors[obs]
        if not full and cumsum_prob < 0.9: #Certain prob. threshold
          continue
        children.append( Node(self.bamdp,obs,new_bel,self.depth+1,self,a,prob,r) )

    return children

from queue import Queue
class TreeSearch:

  def __init__(self,bamdp,Node=None,priorityqueue=None, ignore_leafs = True):
    #Priortyqueue for complicated search, now initiazing with simple queue (BFS)
    self.priorityqueue = priorityqueue
    if priorityqueue == None:
        self.priorityqueue = Queue()
        self.priorityqueue.put(Node)
    self.leafnodes = set() #Need leafnodes for backward induction
    self.bamdp = bamdp
    self.mdpoptimalvalues = {} # V[mdp] = {s1: , s2:  }
    self.ignore_leafs = ignore_leafs
    self.calc_grad = False



  #Main tree expansion logic
  def search(self):
    while not self.priorityqueue.empty():
#      pdb.set_trace()
      curr_node = self.priorityqueue.get()
      if curr_node.depth >= self.bamdp.horizon: #Initilazing if leaf nodes & keeping track
        curr_node.otherdata['V'] = self._init_leafs(curr_node)
        self.leafnodes.add(curr_node)

      children = curr_node.Expand()
      [self.priorityqueue.put(child) for child in children]



  #Value Iteration / Backward Induction
  def Eval_VI(self):
    
    
    if not self.leafnodes:
      print("You didn't generate the tree first\n")
      exit(-1)

    gamma = self.bamdp.gamma
    #Backward Induction
    leafnodes = self.leafnodes
    for depth in range(self.bamdp.horizon-1,-1,-1):

      #Computing Q-Values
      child_nodes = leafnodes
      Q = collections.defaultdict(float)
      for child in child_nodes:
        Q[child.parent,child.act] += child.prob * (child.r + gamma*child.otherdata['V'])

      #Assigining V values
      parents = set()
      [parents.add(node.parent) for node in child_nodes]
      for parent in parents:
        lis = []
        for act in self.bamdp.actions:
          lis.append( (Q[parent,act],act) )
        max_element = max(lis,key=lambda x: x[0])
        parent.otherdata['V'] = max_element[0]
        parent.otherdata['act'] = max_element[1]

      leafnodes = parents
    return None

  def Eval_VI_mdp(self):
    assert self.ignore_leafs
    #Need to fix VI for this also otherwise.

    self.calc_grad = True
    gamma = self.bamdp.gamma

    leafnodes = self.leafnodes
    for depth in range(self.bamdp.horizon-1,-1,-1):
      child_nodes = leafnodes
      for child in child_nodes:
        for mdp in self.bamdp.mdpList:
          if child.parent.otherdata['act'] == child.act:
            child.parent.otherdata["Vmdp"][mdp] += self.bamdp.transitions[mdp][(child.parent.s, child.act)][child.s] * \
                                                   (child.r + gamma*child.otherdata["Vmdp"][mdp])
      parents = set()
      [parents.add(node.parent) for node in child_nodes]
      leafnodes = parents


  #UppeBound leaf initialization
  def _init_leafs(self,node):
    #mdps = random.choices(self.bamdp.mdpList,node.bel,k=10) Shouldn't do this, directly take mdpList
    V = 0.0
    for mdp in self.bamdp.mdpList:
      if mdp in self.mdpoptimalvalues:
        V += self.mdpoptimalvalues[mdp][0][node.s] * node.bel[mdp]
      else:
        if self.ignore_leafs:
          self.mdpoptimalvalues[mdp] = [collections.defaultdict(float), None, None]
        else:
          self.mdpoptimalvalues[mdp] = valueIteration(self.bamdp,mdp)
        V += self.mdpoptimalvalues[mdp][0][node.s] * node.bel[mdp]
          
    return V

  #Compute gradients given a node (which contains its Bayes-opt action info) OR
  #just belief (needed if tree is sampled and some of reachable beliefs non existant is the built tree)
  #ON the hindsight, even tracked belief isn't needed since we need partial derivaties wrt to each mdp
  #ON the hindsight, if a belief-node doesn't exist, I can compute derivative on any subtree starting from it, since I don't know the Bayes-opt action starting from it
  def get_gradient(self,node):#,s=None,belief=None):
    if not self.calc_grad:
      self.Eval_VI_mdp()

    return node.otherdata["Vmdp"]

#Need MDP representation without terminal states (I Guess)
#Doesn't support missing keys, i.e. all keys, even zero-prob transitions must exist
## States,Actions,MDPs can be arbitary objects, as only dictionary datastruct is used by me
if __name__ == "__main__":
  states = range(3)
  actions = range(2)
  discount = 0.5 #horizon = 2
  mdpList = range(2)
  rewards = {}
  rewards[0] = {(0,0) : 0.0,(0,1) : 0.0 , (1,0):1.0 , (2,0):0 , (1,1):0.0 , (2,1):0.0}
  rewards[1] = {(0,0) : 0.0,(0,1) : 0.0 , (1,0):0.0 , (2,0):0 , (1,1):0.0 , (2,1):1.0}
  transitions = {}
  transitions[0] = {(0,0) : {0:0,1:1.0,2:0.0}, (0,1) : {0:0.0,1:0.0,2:1.0}, 
             (1,0): {0:0,1:1.0,2:0.0}, (1,1) : {0:0,1:1.0,2:0.0},
             (2,0): {0:0.0,1:0.0,2:1.0} ,(2,1):{0:0.0,1:0.0,2:1.0}}

  transitions[1] = {(0,0) : {0:0.5,1:0.5,2:0.0}, (0,1) : {0:0.5,1:0.0,2:0.5}, 
             (1,0): {0:0,1:0.5,2:0.5}, (1,1) : {0:0,1:1.0,2:0.0},
             (2,0): {0:0.0,1:0.0,2:1.0} ,(2,1):{0:0.0,1:0.0,2:1.0}}
  
  belief = {0:0.5,1:0.5}

  bamdp = BAMDP(states,actions,rewards,transitions,mdpList,discount)
  print(bamdp.getMarginalR(belief,(1,0)))
  print(bamdp.getMarginalP(belief,(1,0)))
  print(bamdp.getPosterior(belief,(1,0),1))
  
  node = Node(bamdp,0,belief)
  tree = TreeSearch(bamdp,node)
  tree.search()
  tree.Eval_VI()
  print(node.otherdata)
