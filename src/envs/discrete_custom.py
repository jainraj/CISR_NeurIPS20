from gym.envs.toy_text import discrete
import numpy as np

__all__ = ['DiscreteEnvCustom']


class DiscreteEnvCustom(discrete.DiscreteEnv):

    """
    Modfication of the original discrete env to return the info of transitions
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary over states containinng dictionaries over actions. Each one contains
      P[s][a] == {'p': [probability], 'transition': [(next_s, reward, done, info)]} The value of 'p' is a list with the
      probability over the transitions. Transition is a list of tuples. each tuple contains the full info that is the
      output of an openai gym.step env.
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd, timeout=np.inf):
        self.nsteps = 0
        self.timeout = timeout
        super(DiscreteEnvCustom, self).__init__(nS, nA, P, isd)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, info = transitions[i]
        self.s = s
        self.lastaction = a
        info.update({"prob": p})
        self.nsteps += 1
        d = np.logical_or(d, self.nsteps > self.timeout)
        return s, r, d, info

    def reset(self):
        self.nsteps = 0
        return super(DiscreteEnvCustom, self).reset()