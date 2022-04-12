import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
from contextlib import closing
from io import StringIO
from src.envs.discrete_custom import DiscreteEnvCustom
from src.envs.dangerous_grid_world.dangerous_grid_world_constants import UP, DOWN, LEFT, RIGHT, DANGER_STATES, DANGER, SAFE, TERMINAL


class DangerousGridWorldEnvCustom(DiscreteEnvCustom):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self,
                 grid_specs=(8, 8),
                 start_state=(0, 0),
                 end_state=(7, 7),
                 danger_states=DANGER_STATES,
                 timeout=np.inf):

        self.grid_specs = grid_specs
        self.start_state = start_state
        self.end_state = end_state
        self.start_state_index = np.ravel_multi_index(self.start_state, self.grid_specs)

        nS = np.prod(self.grid_specs)
        nA = 4

        # Danger states
        self.danger_states = np.asarray(danger_states)
        # Make sure the start state and end state are always safe
        self.danger_states[start_state] = False
        self.danger_states[end_state] = False

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.grid_specs)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(DangerousGridWorldEnvCustom, self).__init__(nS, nA, P, isd, timeout)

        unique_states = [SAFE, TERMINAL, DANGER, 'agent']
        values = [0., 0.25, 0.5, 0.75]
        self.num_desc = np.zeros_like(self.danger_states, dtype=np.float)
        self.num_desc[self.danger_states] = 0.5
        self.num_desc[end_state] = 0.25

        n = self.observation_space.n
        shape = (int(np.sqrt(n)), int(np.sqrt(n)), 1)  # Last dim for CNN
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=shape, dtype=np.float)
        self.fig, self.ax = (None, None)

    def compute_obs(self):
        new_obs = np.copy(self.num_desc)
        new_obs[np.unravel_index(self.s, self.num_desc.shape)] = 1.0
        return new_obs[:, :, None]

    def step(self, a):
        _, r, done, info = super(DangerousGridWorldEnvCustom, self).step(a)
        s = self.compute_obs()
        return s, r, done, info

    def reset(self):
        # Set self.s
        super(DangerousGridWorldEnvCustom, self).reset()
        return self.compute_obs()

    def get_state(self):
        return self.s

    def set_state(self, s):
        self.s = s

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.grid_specs[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.grid_specs[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        if current == self.end_state:  # stay in end state
            new_position = self.end_state
            is_done = True
            reward = 0
        else:
            new_position = np.array(current) + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)
            new_position = tuple(new_position)
            is_done = new_position == self.end_state or self.danger_states[new_position]
            if new_position == self.end_state:
                reward = 5
            elif self.danger_states[new_position]:
                reward = 0
            else:
                reward = -0.01

        return [(
            1.0,
            np.ravel_multi_index(new_position, self.grid_specs),
            reward,
            is_done,
            {'next_state_type': self._get_state_type(new_position)}
        )]

    def _get_state_type(self, position):
        if position == self.end_state:
            return TERMINAL
        elif self.danger_states[position]:
            return DANGER
        else:
            return SAFE

    def render(self, mode='human', **kwargs):
        """
        Draw map with agent in it.
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=self.grid_specs)
            self.ax = plt.gca()
        plt.cla()
        self.ax.imshow(self.compute_obs()[:, :, 0])
        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    env = DangerousGridWorldEnvCustom()
    env.render()

    a = 0
    # for i in range(100):
    #     a = env.action_space.sample()
    #     s, r, done, i = env.step(a)
    #     if done:
    #         s = env.reset()
    #     env.render()
