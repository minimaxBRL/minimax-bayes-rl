import numpy as np

class Environment:
    def __init__(self,
                 nS,
                 nA,
                 R,
                 T,
                 max_steps):

        self.nS = nS
        self.nA = nA
        self.R = R
        self.T = T
        self.max_steps = max_steps
        self.t = 0
        self.s = 0

    def step(self, a):

        self.s = np.random.choice(range(self.nS), p=self.T[self.s, a, :])
        r = self.R[self.s, a]

        if self.t < self.max_steps:
            self.t += 1
            return self.s, r, False

        else:
            return self.s, r, True

    def reset(self):

        self.t = 0
        self.s = 0
        return 0

class Chain(Environment):
    def __init__(self):
        nS = 5
        nA = 2
        self.big = 10
        self.small = 2
        self.slip = 0.2
        R = np.zeros((nS, nA))
        T = np.zeros((nS, nA, nS))
        R[0:4, 0] = (1-self.slip) * 0 + self.slip * self.small
        R[0:4, 1] = (1-self.slip) * self.small + self.slip * 0
        R[4, 0] = (1-self.slip) * self.big + self.slip * self.small
        R[4, 1] = (1-self.slip) * self.small + self.slip * self.big

        T[0, 0, 1] = (1-self.slip) * 1
        T[1, 0, 2] = (1-self.slip) * 1
        T[2, 0, 3] = (1-self.slip) * 1
        T[3, 0, 4] = (1-self.slip) * 1
        T[4, 0, 4] = (1-self.slip) * 1

        T[:, 0, 0] = self.slip * 1

        T[0, 1, 1] = self.slip * 1
        T[1, 1, 2] = self.slip * 1
        T[2, 1, 3] = self.slip * 1
        T[3, 1, 4] = self.slip * 1
        T[4, 1, 4] = self.slip * 1

        T[:, 1, 0] = (1-self.slip) * 1
        max_steps = 1000
        super().__init__(nS = nS, nA = nA, R=R, T=T, max_steps=max_steps)

