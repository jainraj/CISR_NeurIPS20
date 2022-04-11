__author__ = 'mainak'

import numpy as np
import sys
from contextlib import closing
from io import StringIO
from src.envs.discrete_custom import DiscreteEnvCustom
from src.envs.dangerous_grid_world.dangerous_grid_world_constants import UP, DOWN, LEFT, RIGHT, DANGER_STATES


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

    def get_state(self):
        return self.s

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
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.grid_specs)
        if self.danger_states[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False, {'next_state_type': 'danger'})]

        is_done = tuple(new_position) == self.end_state
        next_state_type = "terminal" if is_done else "safe"
        return [(1.0, new_state, -1, is_done, {'next_state_type': next_state_type})]

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.grid_specs)
            if self.s == s:
                output = " X "  # Current
            # Print terminal state
            elif position == self.end_state:
                output = " T "  # Terminal
            elif self.danger_states[position]:
                output = " D "  # Danger
            else:
                output = " O "  # Safe

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.grid_specs[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
