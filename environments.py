from collections import defaultdict
import  copy
import  numpy as np
def get_environment(env, dim=None, seed=None):

    if env == "test":
        raise NotImplementedError
        #Need to update belief on reward.
        dim = 2
        states = 3
        actions = 2
        mdpList = range(dim+1)
        rewards = {}
        rewards[0] = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 1.0, (2, 0): 0, (1, 1): 0.0, (2, 1): 0.0}
        rewards[1] = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (2, 0): 0, (1, 1): 0.0, (2, 1): 1.0}
        rewards[2] = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.5, (2, 0): 0, (1, 1): 0.0, (2, 1): 0.5}

        transitions = {}
        transitions[0] = {(0, 0): {0: 0, 1: 1.0, 2: 0.0}, (0, 1): {0: 0.0, 1: 0.0, 2: 1.0},
                          (1, 0): {0: 0, 1: 1.0, 2: 0.0}, (1, 1): {0: 0, 1: 1.0, 2: 0.0},
                          (2, 0): {0: 0.0, 1: 0.0, 2: 1.0}, (2, 1): {0: 0.0, 1: 0.0, 2: 1.0}}

        transitions[1] = {(0, 0): {0: 0.5, 1: 0.5, 2: 0.0}, (0, 1): {0: 0.5, 1: 0.0, 2: 0.5},
                          (1, 0): {0: 0, 1: 0.5, 2: 0.5}, (1, 1): {0: 0, 1: 1.0, 2: 0.0},
                          (2, 0): {0: 0.0, 1: 0.0, 2: 1.0}, (2, 1): {0: 0.0, 1: 0.0, 2: 1.0}}
        transitions[2] = {(0, 0): {0: 0.5, 1: 0.5, 2: 0.0}, (0, 1): {0: 0.5, 1: 0.0, 2: 0.5},
                          (1, 0): {0: 0, 1: 0.5, 2: 0.5}, (1, 1): {0: 0, 1: 1.0, 2: 0.0},
                          (2, 0): {0: 0.0, 1: 0.0, 2: 1.0}, (2, 1): {0: 0.0, 1: 0.0, 2: 1.0}}
    elif env == "binary_tree":
        #Need to update belief on reward.
        depth = 3
        dim = 1
        states = int((1-2**(depth+1))/(1-2) + 2)
        actions = 2
        mdpList = range(dim+1)
        rewards = {}
        rewards[0] = defaultdict(int)
        rewards[1] = defaultdict(int)
        rewards[2] = defaultdict(int)

        transitions = {}
        transitions[0] = defaultdict(lambda: defaultdict(int))
        child = 1
        for i in range(depth):
            for j in range(2**i):

                s = 2**i + j -1
                transitions[0][(s, 0)][child] = 1.
                transitions[0][(s, 1)][child+1] = 1.
                child += 2

        for i in mdpList:
            transitions[i] = copy.deepcopy(transitions[0])


        transitions[0][(7,0)][15] = 1
        transitions[0][(14, 1)][16] = 1
        transitions[1][(7,0)][16] = 1
        transitions[1][(14, 1)][15] = 1


        rewards[0][(16,0)] = 1
        rewards[0][(16, 1)] = 1
        rewards[0][(15, 0)] = 0.95
        rewards[0][(15, 1)] = 0.95

        rewards[1][(16,0)] = 1
        rewards[1][(16, 1)] = 1
        rewards[1][(15, 0)] = 0.95
        rewards[1][(15, 1)] = 0.95


    elif env == "chain":
        #Need to update belief on reward.


        dim = 1
        states = 3
        actions = 2
        mdpList = range(dim + 1)
        rewards = {}
        rewards[0] = defaultdict(int)
        rewards[1] = defaultdict(int)
        rewards[2] = defaultdict(int)

        transitions = {}
        transitions[0] = defaultdict(lambda: defaultdict(int))
        transitions[1] = defaultdict(lambda: defaultdict(int))
        transitions[2] = defaultdict(lambda: defaultdict(int))

        #  1 <-> 0 <-> 2,


        #action 0 tries to go left.
        transitions[0][(0,0)][1] = 0.95
        transitions[0][(0,0)][2] = 0.05
        transitions[0][(0,1)][1] = 0.05
        transitions[0][(0,1)][2] = 0.95
        transitions[0][(1, 0)][1] = 0.95
        transitions[0][(1, 0)][0] = 0.05
        transitions[0][(1, 1)][1] = 0.05
        transitions[0][(1, 1)][0] = 0.95
        transitions[0][(2,0)][2] = 0.05
        transitions[0][(2,0)][0] = 0.95
        transitions[0][(2,1)][2] = 0.95
        transitions[0][(2,1)][0] = 0.05


        #action 1 tries to go left.
        transitions[1][(0,0)][1] = 0.05
        transitions[1][(0,0)][2] = 0.95
        transitions[1][(0,1)][1] = 0.95
        transitions[1][(0,1)][2] = 0.05
        transitions[1][(1, 0)][1] = 0.05
        transitions[1][(1, 0)][0] = 0.95
        transitions[1][(1, 1)][1] = 0.95
        transitions[1][(1, 1)][0] = 0.05
        transitions[1][(2,0)][2] = 0.95
        transitions[1][(2,0)][0] = 0.05
        transitions[1][(2,1)][2] = 0.05
        transitions[1][(2,1)][0] = 0.95


        #action 0 tries to go left.
        transitions[2][(0, 0)][1] = 0.5
        transitions[2][(0, 0)][2] = 0.5
        transitions[2][(0, 1)][1] = 0.5
        transitions[2][(0, 1)][2] = 0.5
        transitions[2][(1, 0)][1] = 0.95
        transitions[2][(1, 0)][0] = 0.05
        transitions[2][(1, 1)][1] = 0.05
        transitions[2][(1, 1)][0] = 0.95
        transitions[2][(2, 0)][2] = 0.05
        transitions[2][(2, 0)][0] = 0.95
        transitions[2][(2, 1)][2] = 0.95
        transitions[2][(2, 1)][0] = 0.05



        rewards[0][(1, 0)] = 1
        rewards[0][(1, 1)] = 1

        rewards[1][(1, 0)] = 1
        rewards[1][(1, 1)] = 1

        rewards[2][(1, 0)] = 1
        rewards[2][(1, 1)] = 1

        rewards[0][(2, 0)] = 0.1
        rewards[0][(2, 1)] = 0.1

        rewards[1][(2, 0)] = 0.1
        rewards[1][(2, 1)] = 0.1

        rewards[2][(2, 0)] = 0.1
        rewards[2][(2, 1)] = 0.1


    elif env == "chain2":
        #Need to update belief on reward.


        dim = 2
        states = 3
        actions = 2
        mdpList = range(dim + 1)
        rewards = {}
        rewards[0] = defaultdict(int)
        rewards[1] = defaultdict(int)
        rewards[2] = defaultdict(int)

        transitions = {}
        transitions[0] = defaultdict(lambda: defaultdict(int))
        transitions[1] = defaultdict(lambda: defaultdict(int))
        transitions[2] = defaultdict(lambda: defaultdict(int))

        #  1 <-> 0 <-> 2,


        #action 0 tries to go left.
        transitions[0][(0,0)][1] = 0.95
        transitions[0][(0,0)][2] = 0.05
        transitions[0][(0,1)][1] = 0.05
        transitions[0][(0,1)][2] = 0.95
        transitions[0][(1, 0)][1] = 0.95
        transitions[0][(1, 0)][0] = 0.05
        transitions[0][(1, 1)][1] = 0.05
        transitions[0][(1, 1)][0] = 0.95
        transitions[0][(2,0)][2] = 0.05
        transitions[0][(2,0)][0] = 0.95
        transitions[0][(2,1)][2] = 0.95
        transitions[0][(2,1)][0] = 0.05


        #action 1 tries to go left.
        transitions[1][(0,0)][1] = 0.05
        transitions[1][(0,0)][2] = 0.95
        transitions[1][(0,1)][1] = 0.95
        transitions[1][(0,1)][2] = 0.05
        transitions[1][(1, 0)][1] = 0.05
        transitions[1][(1, 0)][0] = 0.95
        transitions[1][(1, 1)][1] = 0.95
        transitions[1][(1, 1)][0] = 0.05
        transitions[1][(2,0)][2] = 0.95
        transitions[1][(2,0)][0] = 0.05
        transitions[1][(2,1)][2] = 0.05
        transitions[1][(2,1)][0] = 0.95


        #action 0 tries to go left.
        transitions[2][(0, 0)][1] = 0.5
        transitions[2][(0, 0)][2] = 0.5
        transitions[2][(0, 1)][1] = 0.5
        transitions[2][(0, 1)][2] = 0.5
        transitions[2][(1, 0)][1] = 0.95
        transitions[2][(1, 0)][0] = 0.05
        transitions[2][(1, 1)][1] = 0.05
        transitions[2][(1, 1)][0] = 0.95
        transitions[2][(2, 0)][2] = 0.05
        transitions[2][(2, 0)][0] = 0.95
        transitions[2][(2, 1)][2] = 0.95
        transitions[2][(2, 1)][0] = 0.05



        rewards[0][(1, 0)] = 1
        rewards[0][(1, 1)] = 1

        rewards[1][(1, 0)] = 1
        rewards[1][(1, 1)] = 1

        rewards[2][(1, 0)] = 1
        rewards[2][(1, 1)] = 1

        rewards[0][(2, 0)] = 0.1
        rewards[0][(2, 1)] = 0.1

        rewards[1][(2, 0)] = 0.1
        rewards[1][(2, 1)] = 0.1

        rewards[2][(2, 0)] = 0.1
        rewards[2][(2, 1)] = 0.1

    elif env == "jaksch":
        depth = 3
        dim = 3
        states = 3
        actions = 8
        mdpList = range(dim + 1)
        rewards = {}
        rewards[0] = defaultdict(int)
        rewards[1] = defaultdict(int)
        rewards[2] = defaultdict(int)

        transitions = {}
        transitions[0] = defaultdict(lambda: defaultdict(int))
        transitions[1] = defaultdict(lambda: defaultdict(int))
        transitions[2] = defaultdict(lambda: defaultdict(int))

        #  1 <-> 0 <-> 2, action 0 tries to go left.

        delta = 0.1
        epsilon = 0.05



        for i in range(mdpList):
            for a in range(actions):
                transitions[i][(0,a)][0] = 1-delta
                transitions[i][(0, a)][1] = delta
                transitions[i][(1,a)][1] = 1-delta
                transitions[i][(1, a)][0] = delta
                rewards[i][(0, a)] = 0
                rewards[i][(1, a)] = 1

            transitions[i][(0,i)][0] = 1-delta-epsilon
            transitions[i][(0,i)][1] = delta+epsilon
            transitions[i][(1,i)][0] = delta+epsilon
            transitions[i][(1,i)][1] = 1-delta-epsilon


    elif env == "random":
        if dim is None:
            dim = 3
        if seed is None:
            seed = 123
        np.random.seed(seed)
        states = 5
        actions = 3
        mdpList = range(dim + 1)
        rewards = {}
        for i in mdpList:
            rewards[i] = defaultdict(int)
            rewards[i][(states-1, 0)] = 1



        transitions = {}
        for i in mdpList:
            transitions[i] = {}
            for s in range(states):
                for a in range(actions):
                    transitions[i][(s,a)] = {}
                    x = np.random.exponential(1,size=states)
                    x= x/np.sum(x)
                    for snext in range(states):
                        transitions[i][(s,a)][snext] = x[snext]







    return dim, states, actions, mdpList, rewards, transitions
